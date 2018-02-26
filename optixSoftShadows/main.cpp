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
const char *mainPTX, *blurPTX, *parallelogramPTX;

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

ParallelogramLight light;
Buffer light_buffer;

//--------------------------------------------------------------
// Render loop
//--------------------------------------------------------------

void glutDisplay()
{
	updateCamera();

	light.corner = make_float3(343.0f + cos(glutGet(GLUT_ELAPSED_TIME) / 1000.f) * 100.f,
							   548.6f,
							   227.0f + sin(glutGet(GLUT_ELAPSED_TIME) / 1000.f) * 100.f);
	memcpy(light_buffer->map(), &light, sizeof(light));
	light_buffer->unmap();
	context["lights"]->setBuffer(light_buffer);

	context->launch(0, width, height);
	context->launch(1, width, height);
	
	Buffer buffer = context["blur_output"]->getBuffer();
	sutil::displayBufferGL(buffer);

	/*RTsize w, h;
	buffer->getSize(w, h);
	RTsize byteSize = w * h * buffer->getElementSize();

	float *output = reinterpret_cast<float*>(new char[byteSize]);
	memcpy(output, buffer->map(), byteSize);
	buffer->unmap();

	float minval = 10000.0f;
	float maxval = -10000.0f;
	for(int i = 0; i < w*h; i += 4) {
		minval = min(output[i+3], minval);
		maxval = max(output[i+3], maxval);
	}

	printf("max: %f\nmin: %f", maxval, minval);

	delete[] output;*/

	static unsigned frame_count = 0;
	sutil::displayFps(frame_count++);
	sutil::displayText("SoftShadows", 10, height - 15);

	{
		std::stringstream msg;
		msg << "Yaw: " << camera.yaw;
		sutil::displayText(msg.str().c_str(), 10, height - 35);
	}

	{
		std::stringstream msg;
		msg << "Pitch: " << camera.pitch;
		sutil::displayText(msg.str().c_str(), 10, height - 55);
	}

	{
		std::stringstream msg;
		msg << "Position: " << camera.position;
		sutil::displayText(msg.str().c_str(), 10, height - 75);
	}
	
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

	light_buffer = context->createBuffer(RT_BUFFER_INPUT);
	light_buffer->setFormat(RT_FORMAT_USER);
	light_buffer->setElementSize(sizeof(ParallelogramLight));
	light_buffer->setSize(1u);
	memcpy(light_buffer->map(), &light, sizeof(light));
	light_buffer->unmap();
	context["lights"]->setBuffer(light_buffer);

	// Material
	Material diffuse = context->createMaterial();
	diffuse->setClosestHitProgram(0, context->createProgramFromPTXString(mainPTX, "diffuse"));
	diffuse->setAnyHitProgram(1, context->createProgramFromPTXString(mainPTX, "shadow"));

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
		context->setEntryPointCount(2);

		// Load CUDA programs
		mainPTX = loadCudaFile("main.cu");
		blurPTX = loadCudaFile("gaussian_blur.cu");
		parallelogramPTX = loadCudaFile("parallelogram.cu");

		// Create output buffer
		Buffer mainProgramBuffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, true);
		Buffer blurProgramBuffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, true);

		// Set ray generation program
		context->setRayGenerationProgram(0, context->createProgramFromPTXString(mainPTX, "trace_ray"));
		context["main_output"]->set(mainProgramBuffer);

		// Exception program
		context->setExceptionProgram(0, context->createProgramFromPTXString(mainPTX, "exception"));
		context["bad_color"]->setFloat(1.0f, 0.0f, 1.0f);

		// Miss program
		context->setMissProgram(0, context->createProgramFromPTXString(mainPTX, "miss"));
		context["bg_color"]->setFloat(make_float3(0.34f, 0.55f, 0.85f));

		// Set blur program
		context->setRayGenerationProgram(1, context->createProgramFromPTXString(blurPTX, "blur"));
		context["blur_output"]->set(blurProgramBuffer);

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
