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
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sutil.h>

#include "util.h"

using namespace optix;

Context context = 0;
const int width = 1280, height = 720;
float time = 0.0f;

void glutDisplay()
{
	context["time"]->setFloat(time);
	context->launch(0, width, height);
	
	sutil::displayBufferGL(context["output_buffer"]->getBuffer());

	{
		static unsigned frame_count = 0;
		sutil::displayFps(frame_count++);
		sutil::displayText("SoftShadows", 10, height - 15);
	}

	time += 1.0f / 30.0f;

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

void destroyContext()
{
	if(context)
	{
		context->destroy();
		context = 0;
	}
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
		context["output_buffer"]->set(buffer);
		context["time"]->setFloat(time);

		// Set ray generation program
		const char *ptx = loadCudaFile("main.cu");
		context->setRayGenerationProgram(0, context->createProgramFromPTXString(ptx, "trace_ray"));

		context->validate();

		glOrtho(0, 1, 0, 1, -1, 1);

		glutDisplayFunc(glutDisplay);
		glutIdleFunc(glutDisplay);
		glutCloseFunc(destroyContext);
		glutMainLoop();
	} SUTIL_CATCH(context->get())
}