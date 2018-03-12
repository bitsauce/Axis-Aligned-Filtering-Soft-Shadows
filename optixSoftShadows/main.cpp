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

#if defined(__APPLE__)
#  include <GLUT/glut.h>
#else
#  include <GL/glew.h>
#  if defined(_WIN32)
#    include <GL/wglew.h>
#    include <GL/freeglut.h>
#  endif
#  include <GL/glut.h>
#endif

#include <optix.h>
#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sutil.h>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <list>

#include "util.h"
#include "structs.h"

using namespace optix;

// Some forward declarations
void updateCamera();
void initWindow(int*, char**);
void destroyContext();

// OptiX context
Context context = 0;
const int width = 1280, height = 720;
const char *mainPTX, *blurPTX, *parallelogramPTX, *normalizePTX;

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
	BLUR_H_PROGRAM,
	BLUR_V_PROGRAM,
	NORMALIZE_PROGRAM
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

// CUDA buffers
ParallelogramLight light;
Buffer lightBuffer;
Buffer diffuseBuffer;
Buffer depthBuffer;
Buffer objectIdBuffer;
Buffer projectedDistancesBuffer;
Buffer betaBuffer;
Buffer blurHBuffer;
Buffer blurVBuffer;

//--------------------------------------------------------------
// Render loop
//--------------------------------------------------------------

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

void normalizeAndDisplayBuffer(Buffer buffer)
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
}

void glutDisplay()
{
	updateCamera();
	if(animateLight)
	{
		light.corner = make_float3(343.0f + cos(glutGet(GLUT_ELAPSED_TIME) / 1000.f) * 100.f,
								   548.6f,
								   227.0f + sin(glutGet(GLUT_ELAPSED_TIME) / 1000.f) * 100.f);
		memcpy(lightBuffer->map(), &light, sizeof(light));
		lightBuffer->unmap();
		context["lights"]->setBuffer(lightBuffer);
	}

	// Render diffuse image
	context->launch(DIFFUSE_PROGRAM, width, height);

	context["blur_h_buffer"]->set(diffuseBuffer);

	switch(state)
	{
		case DEFAULT:
		{
			// Gaussian blur
			context->launch(BLUR_H_PROGRAM, width, height);
			context->launch(BLUR_V_PROGRAM, width, height);
			sutil::displayBufferGL(context["blur_v_buffer"]->getBuffer());
		}
		break;

		case SHOW_DIFFUSE:
		{
			sutil::displayBufferGL(context["diffuse_buffer"]->getBuffer());
		}
		break;

		case SHOW_DEPTH:
		{
			normalizeAndDisplayBuffer(context["depth_buffer"]->getBuffer());
		}
		break;

		case SHOW_OBJECT_IDS:
		{
			normalizeAndDisplayBuffer(context["object_id_buffer"]->getBuffer());
		}
		break;

		case SHOW_H_BLUR:
		{
			context->launch(BLUR_H_PROGRAM, width, height);
			sutil::displayBufferGL(context["blur_h_buffer"]->getBuffer());
		}
		break;

		case SHOW_V_BLUR:
		{
			context["blur_h_buffer"]->set(diffuseBuffer);
			context->launch(BLUR_V_PROGRAM, width, height);
			sutil::displayBufferGL(context["blur_v_buffer"]->getBuffer());
		}
		break;

		case SHOW_BETA:
		{
			normalizeAndDisplayBuffer(context["beta_buffer"]->getBuffer());
		}
		break;

		/*case SHOW_NUM_SAMPLES:
		{
			// Normalize and display the adaptive sampling buffer
			float minValue, maxValue, avg;
			Buffer buffer = context["num_samples_buffer"]->getBuffer();
			getBufferMinMax(buffer, minValue, maxValue, avg);
			context["max_value"]->setFloat(maxValue);
			context["normalize_buffer"]->set(buffer);
			context->launch(NORMALIZE_PROGRAM, width, height);
			sutil::displayBufferGL(buffer);

			std::stringstream msg;
			msg << "Max samples: " << maxValue;
			sutil::displayText(msg.str().c_str(), width - 200, height - 35);
		}
		break;*/

		default:
			sutil::displayBufferGL(context["diffuse_buffer"]->getBuffer());
			break;
	}

	std::string stateName = "MISSING";
	switch(state)
	{
		case DEFAULT: stateName = "SoftShadows"; break;
		case SHOW_DIFFUSE: stateName = "Diffuse"; break;
		case SHOW_DEPTH: stateName = "Depth"; break;
		case SHOW_OBJECT_IDS: stateName = "Object IDs"; break;
		case SHOW_H_BLUR: stateName = "Blur H"; break;
		case SHOW_V_BLUR: stateName = "Blur V"; break;
		case SHOW_BETA: stateName = "Beta"; break;
		case SHOW_NUM_SAMPLES: stateName = "Num Samples"; break;
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
	topRightInfo.push_back("1/2: Next/Prev State");
	drawStrings(topRightInfo, width - 200, height - 15, 0, -20);

	glutSwapBuffers();
}

//--------------------------------------------------------------
// Scene & geometry
//--------------------------------------------------------------

void setMaterial(
	GeometryInstance& gi,
	Material material,
	const std::string& color_name,
	const float3& color)
{
	gi->addMaterial(material);
	gi[color_name]->setFloat(color);
}

uint objectID = 0;

GeometryInstance createParallelogram(
	const float3& anchor,
	const float3& offset1,
	const float3& offset2)
{
	Program pgram_bounding_box = context->createProgramFromPTXString(parallelogramPTX, "bounds");
	Program pgram_intersection = context->createProgramFromPTXString(parallelogramPTX, "intersect");

	Geometry parallelogram = context->createGeometry();
	parallelogram->setPrimitiveCount(1u);
	parallelogram->setIntersectionProgram(pgram_intersection);
	parallelogram->setBoundingBoxProgram(pgram_bounding_box);

	float3 normal = normalize(cross(offset1, offset2));
	float d = dot(normal, anchor);
	float4 plane = make_float4(normal, d);

	float3 v1 = offset1 / dot(offset1, offset1);
	float3 v2 = offset2 / dot(offset2, offset2);

	parallelogram["plane"]->setFloat(plane);
	parallelogram["anchor"]->setFloat(anchor);
	parallelogram["v1"]->setFloat(v1);
	parallelogram["v2"]->setFloat(v2);

	GeometryInstance gi = context->createGeometryInstance();
	gi->setGeometry(parallelogram);
	gi["object_id"]->setUint(++objectID);
	return gi;
}

void createScene()
{
	// Setup light
	light.corner = make_float3(343.0f, 548.6f, 227.0f);
	light.v1 = make_float3(-130.0f, 0.0f, 0.0f);
	light.v2 = make_float3(0.0f, 0.0f, 130.0f);
	light.normal = normalize(cross(light.v1, light.v2));
	light.emission = make_float3(15.0f, 15.0f, 5.0f);

	lightBuffer = context->createBuffer(RT_BUFFER_INPUT);
	lightBuffer->setFormat(RT_FORMAT_USER);
	lightBuffer->setElementSize(sizeof(ParallelogramLight));
	lightBuffer->setSize(1u);
	memcpy(lightBuffer->map(), &light, sizeof(light));
	lightBuffer->unmap();
	context["lights"]->setBuffer(lightBuffer);

	// Material
	Material diffuse = context->createMaterial();
	diffuse->setClosestHitProgram(DIFFUSE_RAY, context->createProgramFromPTXString(mainPTX, "diffuse"));
	diffuse->setAnyHitProgram(SHADOW_RAY, context->createProgramFromPTXString(mainPTX, "shadow"));

	diffuse["Ka"]->setFloat(0.3f, 0.3f, 0.3f);
	diffuse["Kd"]->setFloat(0.6f, 0.7f, 0.8f);
	diffuse["Ks"]->setFloat(0.8f, 0.9f, 0.8f);
	diffuse["phong_exp"]->setFloat(88);
	diffuse["reflectivity_n"]->setFloat(0.2f, 0.2f, 0.2f);

	// create geometry instances
	std::vector<GeometryInstance> gis;

	const float3 white = make_float3(0.8f, 0.8f, 0.8f);
	const float3 green = make_float3(0.05f, 0.8f, 0.05f);
	const float3 red = make_float3(0.8f, 0.05f, 0.05f);
	const float3 light_em = make_float3(15.0f, 15.0f, 5.0f);

	// Floor
	gis.push_back(createParallelogram(make_float3(0.0f, 0.0f, 0.0f),
									  make_float3(0.0f, 0.0f, 559.2f),
									  make_float3(556.0f, 0.0f, 0.0f)));
	setMaterial(gis.back(), diffuse, "diffuse_color", white);

	// Ceiling
	gis.push_back(createParallelogram(make_float3(0.0f, 548.8f, 0.0f),
									  make_float3(556.0f, 0.0f, 0.0f),
									  make_float3(0.0f, 0.0f, 559.2f)));
	setMaterial(gis.back(), diffuse, "diffuse_color", white);

	// Back wall
	gis.push_back(createParallelogram(make_float3(0.0f, 0.0f, 559.2f),
									  make_float3(0.0f, 548.8f, 0.0f),
									  make_float3(556.0f, 0.0f, 0.0f)));
	setMaterial(gis.back(), diffuse, "diffuse_color", white);

	// Right wall
	gis.push_back(createParallelogram(make_float3(0.0f, 0.0f, 0.0f),
									  make_float3(0.0f, 548.8f, 0.0f),
									  make_float3(0.0f, 0.0f, 559.2f)));
	setMaterial(gis.back(), diffuse, "diffuse_color", green);

	// Left wall
	gis.push_back(createParallelogram(make_float3(556.0f, 0.0f, 0.0f),
									  make_float3(0.0f, 0.0f, 559.2f),
									  make_float3(0.0f, 548.8f, 0.0f)));
	setMaterial(gis.back(), diffuse, "diffuse_color", red);

	// Short block
	gis.push_back(createParallelogram(make_float3(130.0f, 165.0f, 65.0f),
									  make_float3(-48.0f, 0.0f, 160.0f),
									  make_float3(160.0f, 0.0f, 49.0f)));
	setMaterial(gis.back(), diffuse, "diffuse_color", white);
	gis.push_back(createParallelogram(make_float3(290.0f, 0.0f, 114.0f),
									  make_float3(0.0f, 165.0f, 0.0f),
									  make_float3(-50.0f, 0.0f, 158.0f)));
	setMaterial(gis.back(), diffuse, "diffuse_color", white);
	gis.push_back(createParallelogram(make_float3(130.0f, 0.0f, 65.0f),
									  make_float3(0.0f, 165.0f, 0.0f),
									  make_float3(160.0f, 0.0f, 49.0f)));
	setMaterial(gis.back(), diffuse, "diffuse_color", white);
	gis.push_back(createParallelogram(make_float3(82.0f, 0.0f, 225.0f),
									  make_float3(0.0f, 165.0f, 0.0f),
									  make_float3(48.0f, 0.0f, -160.0f)));
	setMaterial(gis.back(), diffuse, "diffuse_color", white);
	gis.push_back(createParallelogram(make_float3(240.0f, 0.0f, 272.0f),
									  make_float3(0.0f, 165.0f, 0.0f),
									  make_float3(-158.0f, 0.0f, -47.0f)));
	setMaterial(gis.back(), diffuse, "diffuse_color", white);

	// Tall block
	gis.push_back(createParallelogram(make_float3(423.0f, 330.0f, 247.0f),
									  make_float3(-158.0f, 0.0f, 49.0f),
									  make_float3(49.0f, 0.0f, 159.0f)));
	setMaterial(gis.back(), diffuse, "diffuse_color", white);
	gis.push_back(createParallelogram(make_float3(423.0f, 0.0f, 247.0f),
									  make_float3(0.0f, 330.0f, 0.0f),
									  make_float3(49.0f, 0.0f, 159.0f)));
	setMaterial(gis.back(), diffuse, "diffuse_color", white);
	gis.push_back(createParallelogram(make_float3(472.0f, 0.0f, 406.0f),
									  make_float3(0.0f, 330.0f, 0.0f),
									  make_float3(-158.0f, 0.0f, 50.0f)));
	setMaterial(gis.back(), diffuse, "diffuse_color", white);
	gis.push_back(createParallelogram(make_float3(314.0f, 0.0f, 456.0f),
									  make_float3(0.0f, 330.0f, 0.0f),
									  make_float3(-49.0f, 0.0f, -160.0f)));
	setMaterial(gis.back(), diffuse, "diffuse_color", white);
	gis.push_back(createParallelogram(make_float3(265.0f, 0.0f, 296.0f),
									  make_float3(0.0f, 330.0f, 0.0f),
									  make_float3(158.0f, 0.0f, -49.0f)));
	setMaterial(gis.back(), diffuse, "diffuse_color", white);
	
	// Create geometry group
	GeometryGroup geometry_group = context->createGeometryGroup(gis.begin(), gis.end());
	geometry_group->setAcceleration(context->createAcceleration("NoAccel"));
	context["scene_geometry"]->set(geometry_group);
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
	case 'p': animateLight = !animateLight; break;
	case 'm': showMenus = !showMenus; break;
	case 'o': generateDisparityMap = true; break;
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
		context->setRayTypeCount(2);
		context->setEntryPointCount(4);

		// Load CUDA programs
		mainPTX = loadCudaFile("main.cu");
		blurPTX = loadCudaFile("gaussian_blur.cu");
		parallelogramPTX = loadCudaFile("parallelogram.cu");
		normalizePTX = loadCudaFile("normalize.cu");

		// Create output buffer
		diffuseBuffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, true);
		depthBuffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT, width, height, false);
		projectedDistancesBuffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT2, width, height, true);
		objectIdBuffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT/*UNSIGNED_INT*/, width, height, false);
		betaBuffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT, width, height, false);
		blurHBuffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, true);
		blurVBuffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, true);

		// Set ray generation program
		context->setRayGenerationProgram(DIFFUSE_PROGRAM, context->createProgramFromPTXString(mainPTX, "trace_ray"));
		context["diffuse_buffer"]->set(diffuseBuffer);
		context["beta_buffer"]->set(betaBuffer);
		context["depth_buffer"]->set(depthBuffer);
		context["object_id_buffer"]->set(objectIdBuffer);
		context["projected_distances_buffer"]->set(projectedDistancesBuffer);

		// Exception program
		context->setExceptionProgram(DIFFUSE_PROGRAM, context->createProgramFromPTXString(mainPTX, "exception"));
		context["bad_color"]->setFloat(1.0f, 0.0f, 1.0f);

		// Miss program
		context->setMissProgram(DIFFUSE_RAY, context->createProgramFromPTXString(mainPTX, "miss"));
		context["bg_color"]->setFloat(make_float3(0.34f, 0.55f, 0.85f));

		// Set blur program
		context->setRayGenerationProgram(BLUR_H_PROGRAM, context->createProgramFromPTXString(blurPTX, "blurH"));
		context->setRayGenerationProgram(BLUR_V_PROGRAM, context->createProgramFromPTXString(blurPTX, "blurV"));
		context["blur_h_buffer"]->set(blurHBuffer);
		context["blur_v_buffer"]->set(blurVBuffer);

		// Set normalize program
		context->setRayGenerationProgram(NORMALIZE_PROGRAM, context->createProgramFromPTXString(normalizePTX, "normalize"));
		context["normalize_buffer"]->set(betaBuffer);

		// Setup scene and camera
		createScene();
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
