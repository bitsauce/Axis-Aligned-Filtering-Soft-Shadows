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
#include <Arcball.h>

#include "util.h"

using namespace optix;

Context context = 0;
const int width = 1280, height = 720;

// Camera struct
struct
{
	float3 position;   // Camera position
	float  pitch, yaw; // Camera orientation (pitch and yaw)
} camera;

// Mouse state
int2       mouse_prev_pos;
int        mouse_button;

void updateCamera();

#include <sstream>

void glutDisplay()
{
	float time = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
	context["time"]->setFloat(time);


	updateCamera();

	context->launch(0, width, height);
	
	sutil::displayBufferGL(context["output_buffer"]->getBuffer());

	static unsigned frame_count = 0;
	sutil::displayFps(frame_count++);
	sutil::displayText("SoftShadows", 10, height - 15);

	{
		std::stringstream msg;
		msg << "Yaw: " << camera.yaw;
		sutil::displayText(msg.str().c_str(), 10, height - 25);
	}

	{
		std::stringstream msg;
		msg << "Pitch: " << camera.pitch;
		sutil::displayText(msg.str().c_str(), 10, height - 45);
	}
	
	glutSwapBuffers();
}

void initWindow(int* argc, char** argv)
{
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_ALPHA | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowSize(width, height);
	glutInitWindowPosition(10, 10);
	glutCreateWindow(argv[0]);
}

void createScene()
{
	const char *ptx = loadCudaFile("box.cu");
	Program box_bounds = context->createProgramFromPTXString(ptx, "box_bounds");
	Program box_intersect = context->createProgramFromPTXString(ptx, "box_intersect");

	// Create box
	Geometry box = context->createGeometry();
	box->setPrimitiveCount(1u);
	box->setBoundingBoxProgram(box_bounds);
	box->setIntersectionProgram(box_intersect);
	box["boxmin"]->setFloat(-2.0f, 0.0f, -2.0f);
	box["boxmax"]->setFloat(2.0f, 7.0f, 2.0f);

	// Material
	Material box_material = context->createMaterial();
	const char *ptx2 = loadCudaFile("main.cu");
	box_material->setClosestHitProgram(0, context->createProgramFromPTXString(ptx2, "closest_hit_radiance"));
	
	// Create geometry instances
	std::vector<GeometryInstance> gis;

	// Create geometry instance
	GeometryInstance gi = context->createGeometryInstance();
	gi->setGeometry(box);
	gi->addMaterial(box_material);

	gis.push_back(gi);

	// Create geometry group
	GeometryGroup geometry_group = context->createGeometryGroup(gis.begin(), gis.end());
	geometry_group->setAcceleration(context->createAcceleration("NoAccel"));
	context["scene_geometry"]->set(geometry_group);
}

void setupCamera()
{
	camera.position = make_float3(7.0f, 9.2f, -6.0f);
	camera.pitch = -0.5;
	camera.yaw = 2.5f;
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

void destroyContext()
{
	if(context)
	{
		context->destroy();
		context = 0;
	}
}

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
	camera.yaw   += (mouse_prev_pos.x - x) * 0.001f;
	camera.pitch -= (mouse_prev_pos.y - y) * 0.001f;

	mouse_prev_pos = make_int2(x, y);
}

void glutKeyboardPress(unsigned char k, int x, int y)
{
	bool key_right = false, key_left = false, key_forward = false, key_backward = false;

	switch(k)
	{
	case 'w':
		key_forward = true;
		break;

	case 'a':
		key_left = true;
		break;

	case 's':
		key_backward = true;
		break;

	case 'd':
		key_right = true;
		break;

	case(27): // ESC
	{
		destroyContext();
		exit(0);
	}
	}

	float3 fwd = make_float3(
		cos(camera.pitch) * cos(camera.yaw),
		sin(camera.pitch),
		cos(camera.pitch) * sin(camera.yaw)
	);
	float3 right = normalize(cross(make_float3(0.0f, 1.0f, 0.0f), fwd));
	//float3 up = cross(fwd, right);

	// Move the camera relative to the direction it is facing
	camera.position += right * float((key_right - key_left) * 1.0f);
	//camera.position += up * float((actionState[MOVE_UP] - actionState[MOVE_DOWN]) * moveSpeed);
	camera.position += fwd * float((key_forward - key_backward) * 1.0f);
}


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
		context->setRayTypeCount(1);
		context->setEntryPointCount(1);

		// Create output buffer
		Buffer buffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, true);

		createScene();
		setupCamera();
		updateCamera();

		// Set ray generation program
		const char *ptx = loadCudaFile("main.cu");
		context->setRayGenerationProgram(0, context->createProgramFromPTXString(ptx, "trace_ray"));
		context["output_buffer"]->set(buffer);
		context["time"]->setFloat(glutGet(GLUT_ELAPSED_TIME) / 1000.0f);

		// Exception program
		context->setExceptionProgram(0, context->createProgramFromPTXString(ptx, "exception"));
		context["bad_color"]->setFloat(1.0f, 0.0f, 1.0f);

		// Miss program
		context->setMissProgram(0, context->createProgramFromPTXString(ptx, "miss"));
		context["bg_color"]->setFloat(make_float3(0.34f, 0.55f, 0.85f));

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
		glutMainLoop();
	} SUTIL_CATCH(context->get())
}