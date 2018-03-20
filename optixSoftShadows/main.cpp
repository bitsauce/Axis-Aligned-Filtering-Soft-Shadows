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

#define SCENE_CLASS DefaultScene
//#define SCENE_CLASS GridScene

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
	GEOMETRY_HIT_PROGRAM,
	SAMPLE_DISTANCES_PROGRAM,
	CALCULATE_BETA_PROGRAM,
	BLUR_H_PROGRAM,
	BLUR_V_PROGRAM,
	NORMALIZE_PROGRAM,
	GROUND_TRUTH_PROGRAM,
	DIFFERENCE_PROGRAM,
	NUM_PROGRAMS
};

// Debug visualization state
enum State
{
	DEFAULT,
	SHOW_DIFFUSE,
	SHOW_H_BLUR,
	SHOW_V_BLUR,
	SHOW_D1,
	SHOW_D2_MIN,
	SHOW_D2_MAX,
	SHOW_BETA,
	SHOW_NUM_SAMPLES,
	NUM_STATES
};

// State varaibles
State state = DEFAULT;
bool animateLight = true;
bool showMenus = true;
bool generateDifferenceMap = false;
bool saveScreenshot = false;
Scene *scene = 0;

// CUDA buffers
Buffer diffuseBuffer;
Buffer geometryHitBuffer;
Buffer geometryNormalBuffer;
Buffer ffnormalBuffer;
Buffer differenceBuffer;
Buffer objectIdBuffer;
Buffer projectedDistancesBuffer;
Buffer d1Buffer;
Buffer d2MinBuffer;
Buffer d2MaxBuffer;
Buffer betaBuffer;
Buffer blurHBuffer;
Buffer blurVBuffer;
Buffer numSamplesBuffer;
Buffer heatmapBuffer;

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
		values.push_back(output[i]);
	}
	delete[] output;

	avg = std::accumulate(values.begin(), values.end(), 0.0f) / values.size();
}

Buffer normalizeBuffer(Buffer buffer, bool display = true)
{
	// Normalize and display the beta buffer
	float minValue, maxValue, avgValue;
	getBufferMinMax(buffer, minValue, maxValue, avgValue);
	context["max_value"]->setFloat(maxValue);
	context["normalize_buffer"]->set(buffer);
	context->launch(NORMALIZE_PROGRAM, width, height);
	
	sutil::displayBufferGL(context["heatmap_buffer"]->getBuffer());

	std::vector<std::string> strings;
	strings.push_back("Min: " + std::to_string(minValue));
	strings.push_back("Max: " + std::to_string(maxValue));
	strings.push_back("Avg: " + std::to_string(avgValue));
	drawStrings(strings, width - 150, 55, 0, -20);

	return context["heatmap_buffer"]->getBuffer();
}

void glutDisplay()
{
	updateCamera();
	scene->update();

	// Sample geometry hits
	context->launch(GEOMETRY_HIT_PROGRAM, width, height);

	// Sample distance values (and diffuse color)
	context->launch(SAMPLE_DISTANCES_PROGRAM, width, height);

	// Calculate beta
	context->launch(CALCULATE_BETA_PROGRAM, width, height);

	context["blur_h_buffer"]->set(diffuseBuffer);

	Buffer bufferToDisplay; bool alreadyShown = false;
	switch(generateDifferenceMap ? DEFAULT : state)
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

		case SHOW_D1:
		{
			bufferToDisplay = normalizeBuffer(context["d1_buffer"]->getBuffer());
			alreadyShown = true;
		}
		break;

		case SHOW_D2_MIN:
		{
			bufferToDisplay = normalizeBuffer(context["d2_min_buffer"]->getBuffer());
			alreadyShown = true;
		}
		break;

		case SHOW_D2_MAX:
		{
			bufferToDisplay = normalizeBuffer(context["d2_max_buffer"]->getBuffer());
			alreadyShown = true;
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

	if(generateDifferenceMap)
	{
		// Render ground truth image
		context->launch(GROUND_TRUTH_PROGRAM, width, height);
		
		// Calculate differences between ground truth and filtered image
		context->launch(DIFFERENCE_PROGRAM, width, height);
		
		// Save all three images
		std::string timeStamp = getTimeStamp();
		sutil::displayBufferPPM(("screenshots/" + timeStamp + " filtered.ppm").c_str(), context["blur_v_buffer"]->getBuffer());
		sutil::displayBufferPPM(("screenshots/" + timeStamp + " ground_truth.ppm").c_str(), context["diffuse_buffer"]->getBuffer());
		sutil::displayBufferPPM(("screenshots/" + timeStamp + " difference.ppm").c_str(), context["difference_buffer"]->getBuffer());

		// Toggle difference generation
		generateDifferenceMap = false;
	}
	else
	{
		// Show buffer
		if(!alreadyShown)
		{
			sutil::displayBufferGL(bufferToDisplay);
		}
	}

	std::string stateName = "MISSING";
	switch(state)
	{
		case DEFAULT: stateName = "Soft Shadows"; break;
		case SHOW_DIFFUSE: stateName = "Diffuse"; break;
		case SHOW_H_BLUR: stateName = "Blur H"; break;
		case SHOW_V_BLUR: stateName = "Blur V"; break;
		case SHOW_D1: stateName = "d1"; break;
		case SHOW_D2_MIN: stateName = "d2 min"; break;
		case SHOW_D2_MAX: stateName = "d2 max"; break;
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
	topRightInfo.push_back("O: Generate Diff. Map");
	topRightInfo.push_back("C: Capture Screen");
	topRightInfo.push_back("1/2: Prev/Next State");
	drawStrings(topRightInfo, width - 200, height - 15, 0, -20);

	glutSwapBuffers();
	context->validate();
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
	case 'o': generateDifferenceMap = true; break;
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
		cudaFiles["calculate_difference"] = loadCudaFile("calculate_difference.cu");

		// Create output buffer
		diffuseBuffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT3, width, height, false);
		differenceBuffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT3, width, height, false);
		geometryHitBuffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT3, width, height, false);
		geometryNormalBuffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT3, width, height, false);
		ffnormalBuffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT3, width, height, false);
		projectedDistancesBuffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT2, width, height, false);
		objectIdBuffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT, width, height, false);
		d1Buffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT, width, height, false);
		d2MinBuffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT, width, height, false);
		d2MaxBuffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT, width, height, false);
		betaBuffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT, width, height, false);
		blurHBuffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT3, width, height, false);
		blurVBuffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT3, width, height, false);
		numSamplesBuffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT, width, height, false);
		heatmapBuffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT3, width, height, false);

		// Set ray generation program
		context->setRayGenerationProgram(SAMPLE_DISTANCES_PROGRAM, context->createProgramFromPTXString(cudaFiles["main"], "sample_distances"));
		context["diffuse_buffer"]->set(diffuseBuffer);
		context["beta_buffer"]->set(betaBuffer);
		context["d1_buffer"]->set(d1Buffer);
		context["d2_min_buffer"]->set(d2MinBuffer);
		context["d2_max_buffer"]->set(d2MaxBuffer);
		context["ffnormal_buffer"]->set(ffnormalBuffer);
		context["object_id_buffer"]->set(objectIdBuffer);
		context["projected_distances_buffer"]->set(projectedDistancesBuffer);
		context["num_samples_buffer"]->set(numSamplesBuffer);

		// Set calculate beta program
		context->setRayGenerationProgram(CALCULATE_BETA_PROGRAM, context->createProgramFromPTXString(cudaFiles["main"], "calculate_beta"));

		// Exception program
		context->setExceptionProgram(SAMPLE_DISTANCES_PROGRAM, context->createProgramFromPTXString(cudaFiles["main"], "exception"));
		context["bad_color"]->setFloat(1.0f, 0.0f, 1.0f);
		context["bg_color"]->setFloat(make_float3(0.34f, 0.55f, 0.85f));

		context->setRayGenerationProgram(GEOMETRY_HIT_PROGRAM, context->createProgramFromPTXString(cudaFiles["main"], "trace_primary_ray"));
		context["geometry_hit_buffer"]->setBuffer(geometryHitBuffer);
		context["geometry_normal_buffer"]->setBuffer(geometryNormalBuffer);

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
		context["heatmap_buffer"]->set(heatmapBuffer);

		// Set normalize program
		context->setRayGenerationProgram(DIFFERENCE_PROGRAM, context->createProgramFromPTXString(cudaFiles["calculate_difference"], "calculate_difference"));
		context["difference_buffer"]->set(differenceBuffer);
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
