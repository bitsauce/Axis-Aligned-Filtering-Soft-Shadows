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

//--------------------------------------------------------------
// Per ray data struct
//--------------------------------------------------------------

struct PerRayData_radiance
{
	float3 result;
};

//--------------------------------------------------------------
// Variable declarations
//--------------------------------------------------------------

// Input pixel-coordinate
// An uint2 value (x, y) bound to internal state variable "rtLaunchIndex"
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );

// Output buffer (final image)
// A 2-dimensional buffer of float4s
rtBuffer<float4, 2> output_buffer;

// Shading normal from intersection program
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );

// Scene geometry objects
rtDeclareVariable(rtObject, scene_geometry,,);

// Pinhole camera variables
rtDeclareVariable(float3, eye, , );
rtDeclareVariable(float3, U, , );
rtDeclareVariable(float3, V, , );
rtDeclareVariable(float3, W, , );

//--------------------------------------------------------------
// Main ray program
//--------------------------------------------------------------
RT_PROGRAM void trace_ray()
{
	size_t2 screen = output_buffer.size();
	float2 d = make_float2(launch_index) / make_float2(screen) * 2.f - 1.f; // pixel-coordinate [-1, 1] range
	float3 ray_origin = eye;
	float3 ray_direction = normalize(d.x*U + d.y*V + W);

	// Initialze per-ray data
	PerRayData_radiance prd;

	// Trace the ray in the direction of the camera
	Ray ray = make_Ray(ray_origin, ray_direction, 0, EPSILON, RT_DEFAULT_MAX);
	rtTrace(scene_geometry, ray, prd);

	// Set output color
	output_buffer[launch_index] = make_float4(prd.result, 0.f);
}

//--------------------------------------------------------------
// Closest hit radiance
//--------------------------------------------------------------
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );

RT_PROGRAM void closest_hit_radiance()
{
	prd_radiance.result = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
}

//--------------------------------------------------------------
// Miss program
//--------------------------------------------------------------
rtDeclareVariable(float3, bg_color,,);

RT_PROGRAM void miss()
{
	prd_radiance.result = bg_color;
}

//--------------------------------------------------------------
// Exception
//--------------------------------------------------------------
rtDeclareVariable(float3, bad_color,,);

RT_PROGRAM void exception()
{
	output_buffer[launch_index] = make_float4(bad_color, 1.0f);
}