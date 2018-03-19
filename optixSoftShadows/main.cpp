/*
* Copyright (c) 2017 NVIDIA CORPORATION. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*  * Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
*  * Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
*  * Neither the name of NVIDIA CORPORATION nor the names of its
*    contributors may be used to endorse or promote products derived
*    from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
* PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
* CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
* OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "common.h"
#include "util.h"
#include "structs.h"
#include "scenes.h"

#define SCENE_CLASS GridScene

Context context = 0;
const int width = 1280, height = 720;
std::map<std::string, const char*> cudaFiles;

// Some forward declarations
void updateCamera();
void initWindow(int*, char**);
void destroyContext();

// Camera
struct
{
	float3 position;   // Camera position
	float  pitch, yaw; // Camera orientation (pitch and yaw)
} camera;

const float move_speed = 10.0f;
const float rotation_speed = 0.005f;

// Mouse state
int2       mouse_prev_pos;
int        mouse_button;

// Enum of keyboard input actions
enum Action
{
	MOVE_UP,
	MOVE_DOWN,
	MOVE_LEFT,
	MOVE_RIGHT,
	MOVE_FORWARD,
	MOVE_BACKWARD,
	ACTION_COUNT
};

// Action state list
bool actionState[ACTION_COUNT];

// CUDA programs
enum
{
	DIFFUSE_PROGRAM,
	GEOMETRY_HIT_PROGRAM,
	BLUR_H_PROGRAM,
	BLUR_V_PROGRAM,
	NORMALIZE_PROGRAM,
	GROUND_TRUTH_PROGRAM,
	DISPARITY_PROGRAM,
	NUM_PROGRAMS
};

// Debug visualization state
enum State
{
	DEFAULT,
	SHOW_DIFFUSE,
	SHOW_DEPTH,
	SHOW_OBJECT_IDS,
	SHOW_H_BLUR,
	SHOW_V_BLUR,
	SHOW_BETA,
	SHOW_NUM_SAMPLES,
	NUM_STATES
};

// State varaibles
State state = DEFAULT;
bool animateLight = true;
bool showMenus = true;
bool generateDisparityMap = false;
bool saveScreenshot = false;
Scene *scene = 0;

// CUDA buffers
Buffer diffuseBuffer;
Buffer geometryHitBuffer;
Buffer disparityBuffer;
Buffer depthBuffer;
Buffer objectIdBuffer;
Buffer projectedDistancesBuffer;
Buffer betaBuffer;
Buffer blurHBuffer;
Buffer blurVBuffer;
Buffer numSamplesBuffer;

//--------------------------------------------------------------
// Render loop
//--------------------------------------------------------------

void drawStrings(std::vector<std::string> strings, int x, int y, int dx, int dy)
{
	if(!showMenus) return;
	for(int i = 0; i < strings.size(); i++)
	{
		sutil::displayText(strings[i].c_str(), x, y);
		x += dx;
		y += dy;
	}
}

void getBufferMinMax(Buffer buffer, float &minValue, float &maxValue, float &avg)
{
	// Read CUDA buffer data
	RTsize w, h;
	buffer->getSize(w, h);
	RTsize byteSize = w * h * buffer->getElementSize();
	float *output = reinterpret_cast<float*>(new char[byteSize]);
	memcpy(output, buffer->map(), byteSize);
	buffer->unmap();

	// Extract min and max value
	minValue = FLT_MAX;
	maxValue = -FLT_MAX;
	std::list<float> values;
	for(int i = 0; i < w*h; i++) {
		minValue = std::min(output[i], minValue);
		maxValue = std::max(output[i], maxValue);
		if(output[i] > 0.0f) {
			values.push_back(output[i]);
		}
	}
	delete[] output;

	avg = std::accumulate(values.begin(), values.end(), 0.0f) / values.size();
}

Buffer normalizeBuffer(Buffer buffer)
{
	// Normalize and display the beta buffer
	float minValue, maxValue, avgValue;
	getBufferMinMax(buffer, minValue, maxValue, avgValue);
	context["max_value"]->setFloat(maxValue);
	context["normalize_buffer"]->set(buffer);
	context->launch(NORMALIZE_PROGRAM, width, height);
	
	sutil::displayBufferGL(buffer);

	std::vector<std::string> strings;
	strings.push_back("Min: " + std::to_string(minValue));
	strings.push_back("Max: " + std::to_string(maxValue));
	strings.push_back("Avg: " + std::to_string(avgValue));
	drawStrings(strings, width - 150, 55, 0, -20);

	return buffer;
}

void glutDisplay()
{
	updateCamera();
	scene->update();

	// Sample geometry hits
	context->launch(GEOMETRY_HIT_PROGRAM, width, height);

	// Render diffuse image
	context->launch(DIFFUSE_PROGRAM, width, height);

	context["blur_h_buffer"]->set(diffuseBuffer);

	Buffer bufferToDisplay; bool alreadyShown = false;
	switch(generateDisparityMap ? DEFAULT : state)
	{
		case DEFAULT:
		{
			context->launch(BLUR_H_PROGRAM, width, height);
			context->launch(BLUR_V_PROGRAM, width, height);
			bufferToDisplay = context["blur_v_buffer"]->getBuffer();
		}
		break;

		case SHOW_DIFFUSE:
		{
			bufferToDisplay = context["diffuse_buffer"]->getBuffer();
		}
		break;

		case SHOW_DEPTH:
		{
			bufferToDisplay = normalizeBuffer(context["depth_buffer"]->getBuffer());
			alreadyShown = true;
		}
		break;

		case SHOW_OBJECT_IDS:
		{
			bufferToDisplay = normalizeBuffer(context["object_id_buffer"]->getBuffer());
			alreadyShown = true;
		}
		break;

		case SHOW_H_BLUR:
		{
			context->launch(BLUR_H_PROGRAM, width, height);
			bufferToDisplay = context["blur_h_buffer"]->getBuffer();
		}
		break;

		case SHOW_V_BLUR:
		{
			context["blur_h_buffer"]->set(diffuseBuffer);
			context->launch(BLUR_V_PROGRAM, width, height);
			bufferToDisplay = context["blur_v_buffer"]->getBuffer();
		}
		break;

		case SHOW_BETA:
		{
			bufferToDisplay = normalizeBuffer(context["beta_buffer"]->getBuffer());
			alreadyShown = true;
		}
		break;

		case SHOW_NUM_SAMPLES:
		{
			bufferToDisplay = normalizeBuffer(context["num_samples_buffer"]->getBuffer());
			alreadyShown = true;
		}
		break;

		default:
			bufferToDisplay = context["diffuse_buffer"]->getBuffer();
			break;
	}

	// Show buffer
	if(!alreadyShown) sutil::displayBufferGL(bufferToDisplay);

	if(generateDisparityMap)
	{
		// Render ground truth image
		context->launch(GROUND_TRUTH_PROGRAM, width, height);
		
		// Calculate disparities between ground truth and
		// filtered image
		context->launch(DISPARITY_PROGRAM, width, height);

		// Save all three images
		std::string timeStamp = getTimeStamp();
		sutil::displayBufferPPM(("screenshots/" + timeStamp + " filtered.ppm").c_str(), context["blur_v_buffer"]->getBuffer());
		sutil::displayBufferPPM(("screenshots/" + timeStamp + " ground_truth.ppm").c_str(), context["diffuse_buffer"]->getBuffer());
		sutil::displayBufferPPM(("screenshots/" + timeStamp + " disparity_map.ppm").c_str(), context["disparity_buffer"]->getBuffer());

		// Toggle disparity map generation
		generateDisparityMap = false;
	}

	std::string stateName = "MISSING";
	switch(state)
	{
		case DEFAULT: stateName = "Soft Shadows"; break;
		case SHOW_DIFFUSE: stateName = "Diffuse"; break;
		case SHOW_DEPTH: stateName = "Depth"; break;
		case SHOW_OBJECT_IDS: stateName = "Object IDs"; break;
		case SHOW_H_BLUR: stateName = "Blur H"; break;
		case SHOW_V_BLUR: stateName = "Blur V"; break;
		case SHOW_BETA: stateName = "Beta"; break;
		case SHOW_NUM_SAMPLES: stateName = "Num Samples"; break;
	}

	if(saveScreenshot) 
	{
		std::string timeStamp = getTimeStamp();
		sutil::displayBufferPPM(("screenshots/" + timeStamp + " " + stateName + ".ppm").c_str(), bufferToDisplay);
		saveScreenshot = false;
	}

	// Display world info
	static unsigned frame_count = 0;
	sutil::displayFps(frame_count++);

	std::vector<std::string> topLeftInfo;
	topLeftInfo.push_back(stateName);
	topLeftInfo.push_back("Yaw: " + std::to_string(camera.yaw));
	topLeftInfo.push_back("Pitch: " + std::to_string(camera.pitch));
	topLeftInfo.push_back("Position: [" + std::to_string(camera.position.x) + ", " + std::to_string(camera.position.y) + ", " + std::to_string(camera.position.z) + "]");
	drawStrings(topLeftInfo, 10, height - 15, 0, -20);

	std::vector<std::string> topRightInfo;
	topRightInfo.push_back("WASD: Move");
	topRightInfo.push_back("QE: Up/Down");
	topRightInfo.push_back("P: Pause Animations");
	topRightInfo.push_back("M: Toggle Menus");
	topRightInfo.push_back("O: Generate Disparity Map");
	topRightInfo.push_back("C: Capture Screen");
	topRightInfo.push_back("1/2: Prev/Next State");
	drawStrings(topRightInfo, width - 200, height - 15, 0, -20);

	glutSwapBuffers();
}

//--------------------------------------------------------------
// Camera
//--------------------------------------------------------------

void setupCamera()
{
	camera.position = make_float3(275.0f, 340.0f, -345.0f);
	camera.pitch = 0.0f;
	camera.yaw = 1.5f;
}

void updateCamera()
{
	const float vfov = 60.0f;
	const float aspect_ratio = static_cast<float>(width) / static_cast<float>(height);

	float3 fwd = make_float3(
		cos(camera.pitch) * cos(camera.yaw),
		sin(camera.pitch),
		cos(camera.pitch) * sin(camera.yaw)
	);

	float3 right = normalize(cross(make_float3(0.0f, 1.0f, 0.0f), fwd));
	float3 up = cross(fwd, right);

	// Move the camera relative to the direction it is facing
	camera.position += right * float((actionState[MOVE_LEFT] - actionState[MOVE_RIGHT]) * move_speed);
	camera.position += up * float((actionState[MOVE_UP] - actionState[MOVE_DOWN]) * move_speed);
	camera.position += fwd * float((actionState[MOVE_FORWARD] - actionState[MOVE_BACKWARD]) * move_speed);

	float3 camera_lookat = camera.position + fwd;

	float3 camera_u, camera_v, camera_w;
	sutil::calculateCameraVariables(
		camera.position, camera_lookat, make_float3(0.0f, 1.0f, 0.0f),
		vfov, aspect_ratio,
		camera_u, camera_v, camera_w, true);

	context["eye"]->setFloat(camera.position);
	context["U"]->setFloat(camera_u);
	context["V"]->setFloat(camera_v);
	context["W"]->setFloat(camera_w);
}

//--------------------------------------------------------------
// Input controls
//--------------------------------------------------------------

void glutMousePress(int button, int state, int x, int y)
{
	if(state == GLUT_DOWN)
	{
		mouse_button = button;
		mouse_prev_pos = make_int2(x, y);
	}
	else
	{
		// nothing
	}
}

void glutMouseMotion(int x, int y)
{
	camera.yaw   -= (mouse_prev_pos.x - x) * rotation_speed;
	camera.pitch += (mouse_prev_pos.y - y) * rotation_speed;

	mouse_prev_pos = make_int2(x, y);
}

void glutKeyboardPress(unsigned char k, int, int)
{
	switch(k)
	{
	case 'w': actionState[MOVE_FORWARD] = true; break;
	case 'a': actionState[MOVE_LEFT] = true; break;
	case 's': actionState[MOVE_BACKWARD] = true; break;
	case 'd': actionState[MOVE_RIGHT] = true; break;
	case 'q': actionState[MOVE_UP] = true; break;
	case 'e': actionState[MOVE_DOWN] = true; break;
	case 27: // ESC
	{
		destroyContext();
		exit(0);
	}
	}
}

void glutKeyboardUp(unsigned char k, int, int)
{
	switch( k )
	{
	case 'w': actionState[MOVE_FORWARD] = false; break;
	case 'a': actionState[MOVE_LEFT] = false; break;
	case 's': actionState[MOVE_BACKWARD] = false; break;
	case 'd': actionState[MOVE_RIGHT] = false; break;
	case 'q': actionState[MOVE_UP] = false; break;
	case 'e': actionState[MOVE_DOWN] = false; break;
	case 'p': scene->animate = !scene->animate; break;
	case 'm': showMenus = !showMenus; break;
	case 'o': generateDisparityMap = true; break;
	case 'c': saveScreenshot = true; break;
	case '2': state = State((state + 1) % NUM_STATES); break;
	case '1': state = State((state - 1 + NUM_STATES) % NUM_STATES); break;
	}
}

//--------------------------------------------------------------
// Main
//--------------------------------------------------------------

int main(int argc, char* argv[])
{
	try
	{
		// Init GLUT
		initWindow(&argc, argv);

#ifndef __APPLE__
		glewInit();
#endif

		// Create optix context
		context = Context::create();
		context->setRayTypeCount(NUM_RAYS);
		context->setEntryPointCount(NUM_PROGRAMS);

		// Load CUDA programs
		cudaFiles["main"]          = loadCudaFile("main.cu");
		cudaFiles["ground_truth"]  = loadCudaFile("ground_truth.cu");
		cudaFiles["gaussian_blur"] = loadCudaFile("gaussian_blur.cu");
		cudaFiles["parallelogram"] = loadCudaFile("parallelogram.cu");
		cudaFiles["normalize"]     = loadCudaFile("normalize.cu");
		cudaFiles["calculate_disparity"] = loadCudaFile("calculate_disparity.cu");

		// Create output buffer
		diffuseBuffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, true);
		disparityBuffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, true);
		geometryHitBuffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT3, width, height, false);
		depthBuffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT, width, height, false);
		projectedDistancesBuffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT2, width, height, true);
		objectIdBuffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT/*UNSIGNED_INT*/, width, height, false);
		betaBuffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT, width, height, false);
		blurHBuffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, true);
		blurVBuffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, true);
		numSamplesBuffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT, width, height, false);

		// Set ray generation program
		context->setRayGenerationProgram(DIFFUSE_PROGRAM, context->createProgramFromPTXString(cudaFiles["main"], "trace_ray"));
		context["diffuse_buffer"]->set(diffuseBuffer);
		context["beta_buffer"]->set(betaBuffer);
		context["depth_buffer"]->set(depthBuffer);
		context["object_id_buffer"]->set(objectIdBuffer);
		context["projected_distances_buffer"]->set(projectedDistancesBuffer);
		context["num_samples_buffer"]->set(numSamplesBuffer);

		// Exception program
		context->setExceptionProgram(DIFFUSE_PROGRAM, context->createProgramFromPTXString(cudaFiles["main"], "exception"));
		context["bad_color"]->setFloat(1.0f, 0.0f, 1.0f);

		// Miss program
		context->setMissProgram(DIFFUSE_RAY, context->createProgramFromPTXString(cudaFiles["main"], "miss"));
		context["bg_color"]->setFloat(make_float3(0.34f, 0.55f, 0.85f));


		context->setRayGenerationProgram(GEOMETRY_HIT_PROGRAM, context->createProgramFromPTXString(cudaFiles["main"], "trace_geometry_hit"));
		context["geometry_hit_buffer"]->setBuffer(geometryHitBuffer);

		// Set ray generation program
		context->setRayGenerationProgram(GROUND_TRUTH_PROGRAM, context->createProgramFromPTXString(cudaFiles["ground_truth"], "trace_ray"));
		context->setExceptionProgram(GROUND_TRUTH_PROGRAM, context->createProgramFromPTXString(cudaFiles["ground_truth"], "exception"));
		context->setMissProgram(GROUND_TRUTH_RAY, context->createProgramFromPTXString(cudaFiles["ground_truth"], "miss"));

		// Set blur program
		context->setRayGenerationProgram(BLUR_H_PROGRAM, context->createProgramFromPTXString(cudaFiles["gaussian_blur"], "blurH"));
		context->setRayGenerationProgram(BLUR_V_PROGRAM, context->createProgramFromPTXString(cudaFiles["gaussian_blur"], "blurV"));
		context["blur_h_buffer"]->set(blurHBuffer);
		context["blur_v_buffer"]->set(blurVBuffer);

		// Set normalize program
		context->setRayGenerationProgram(NORMALIZE_PROGRAM, context->createProgramFromPTXString(cudaFiles["normalize"], "normalize"));
		context["normalize_buffer"]->set(betaBuffer);

		// Set normalize program
		context->setRayGenerationProgram(DISPARITY_PROGRAM, context->createProgramFromPTXString(cudaFiles["calculate_disparity"], "calculate_disparity"));
		context["disparity_buffer"]->set(disparityBuffer);
		context["input_buffer_0"]->set(blurVBuffer);
		context["input_buffer_1"]->set(diffuseBuffer);

		// Setup scene and camera
		scene = new SCENE_CLASS();
		setupCamera();
		updateCamera();

		context->validate();

		// Initialize GL state
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(0, 1, 0, 1, -1, 1);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		glViewport(0, 0, width, height);

		glutDisplayFunc(glutDisplay);
		glutIdleFunc(glutDisplay);
		glutCloseFunc(destroyContext);
		glutMotionFunc(glutMouseMotion);
		glutMouseFunc(glutMousePress);
		glutKeyboardFunc(glutKeyboardPress);
		glutKeyboardUpFunc(glutKeyboardUp);
		glutMainLoop();
	} SUTIL_CATCH(context->get())
}

void initWindow(int* argc, char** argv)
{
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_ALPHA | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowSize(width, height);
	glutInitWindowPosition(10, 10);
	glutCreateWindow(argv[0]);
}

void destroyContext()
{
	if(context)
	{
		context->destroy();
		context = 0;
	}
}
