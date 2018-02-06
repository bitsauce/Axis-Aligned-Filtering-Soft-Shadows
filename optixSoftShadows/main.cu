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
#include <optixu/optixu_math_namespace.h>

using namespace optix;

#define EPSILON  1.e-3f

struct PerRayData_pathtrace
{
	float3 result;
	float3 radiance;
	float3 attenuation;
	float3 origin;
	float3 direction;
	unsigned int seed;
	int depth;
	int countEmitted;
	int done;
};

// Input pixel-coordinate
// An uint2 value (x, y) bound to internal
// state variable, rtLaunchIndex
rtDeclareVariable(uint2, launch_index, rtLaunchIndex,);

// Output buffer (final image)
// A 2-dimensional buffer of float4s
rtBuffer<float4, 2> output_buffer;

// Simulation time variable passed from program
rtDeclareVariable(float, time,,);
rtDeclareVariable(rtObject, scene_geometry,,);

rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );

// Main ray program
RT_PROGRAM void trace_ray()
{
	float intensity = fmodf(time, 2.0f);

	float3 ray_origin = make_float3(0.0f);
	float3 ray_direction = make_float3(1.0f, 0.0f, 0.0f);

	// Initialze per-ray data
	PerRayData_pathtrace prd;
	prd.result = make_float3(0.f);
	prd.attenuation = make_float3(1.f);
	prd.countEmitted = true;
	prd.done = false;
	prd.seed = 0;
	prd.depth = 0;

	Ray ray = make_Ray(ray_origin, ray_direction, 0, EPSILON, RT_DEFAULT_MAX);
	rtTrace(scene_geometry, ray, prd);

	output_buffer[launch_index] = make_float4(make_float3(prd.radiance), 0.f);
}


//
// Returns solid color for miss rays
//
rtDeclareVariable(float3, bg_color, , );
RT_PROGRAM void miss()
{
	prd.radiance = bg_color;
}


//
// Returns shading normal as the surface shading result
// 
RT_PROGRAM void closest_hit_radiance0()
{
	prd.radiance.result = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal))*0.5f + 0.5f;
}


//
// Set pixel to solid color upon failur
//
RT_PROGRAM void exception()
{
	output_buffer[launch_index] = make_color(bad_color);
}