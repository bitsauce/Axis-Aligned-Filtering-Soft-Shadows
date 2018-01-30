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

#include <optix.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sutil.h>

#include "util.h"

int main(int argc, char* argv[])
{
	RTcontext context = 0;

	try {
		/* Primary RTAPI objects */
		RTprogram ray_gen_program;
		RTbuffer  buffer;

		/* Parameters */
		RTvariable result_buffer;
		RTvariable draw_color;

		const int width = 1280, height = 720;

		sutil::initGlut(&argc, argv);

		/* Create our objects and set state */
		RT_CHECK_ERROR(rtContextCreate(&context));
		RT_CHECK_ERROR(rtContextSetRayTypeCount(context, 1));
		RT_CHECK_ERROR(rtContextSetEntryPointCount(context, 1));

		/* Set up output buffer (image) */
		RT_CHECK_ERROR(rtBufferCreate(context, RT_BUFFER_OUTPUT, &buffer));
		RT_CHECK_ERROR(rtBufferSetFormat(buffer, RT_FORMAT_FLOAT4));
		RT_CHECK_ERROR(rtBufferSetSize2D(buffer, width, height));
		RT_CHECK_ERROR(rtContextDeclareVariable(context, "result_buffer", &result_buffer));
		RT_CHECK_ERROR(rtVariableSetObject(result_buffer, buffer));

		const char *ptx = loadCudaFile("draw_color.cu");
		RT_CHECK_ERROR(rtProgramCreateFromPTXString(context, ptx, "draw_solid_color", &ray_gen_program));
		RT_CHECK_ERROR(rtProgramDeclareVariable(ray_gen_program, "draw_color", &draw_color));
		//RT_CHECK_ERROR(rtVariableSet3f(draw_color, 0.462f, 0.725f, 0.0f));
		RT_CHECK_ERROR(rtVariableSet3f(draw_color, 1.f, 0.f, 0.0f));
		RT_CHECK_ERROR(rtContextSetRayGenerationProgram(context, 0, ray_gen_program));

		/* Run */
		RT_CHECK_ERROR(rtContextValidate(context));
		RT_CHECK_ERROR(rtContextLaunch2D(context, 0 /* entry point */, width, height));

		/* Display image */
		sutil::displayBufferGlut(argv[0], buffer);

		/* Clean up */
		RT_CHECK_ERROR(rtBufferDestroy(buffer));
		RT_CHECK_ERROR(rtProgramDestroy(ray_gen_program));
		RT_CHECK_ERROR(rtContextDestroy(context));

		return(0);

	} SUTIL_CATCH(context)
}