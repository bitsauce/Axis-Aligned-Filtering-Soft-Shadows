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
//#include "light.h"
#include "random.h"

using namespace optix;

#define EPSILON  1.e-3f

//--------------------------------------------------------------
// Per-ray data structs
//--------------------------------------------------------------

struct PerRayData_radiance
{
	float3 result;
	float  importance;
	int depth;
};

struct PerRayData_shadow
{
	float3 attenuation;
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

// Scene geometry objects
rtDeclareVariable(rtObject, scene_geometry,,);

// Pinhole camera variables
rtDeclareVariable(float3, eye, , );
rtDeclareVariable(float3, U,   , );
rtDeclareVariable(float3, V,   , );
rtDeclareVariable(float3, W,   , );

rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );

rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );

//rtBuffer<ParallelogramLight> lights;

//--------------------------------------------------------------
// Main ray program
//--------------------------------------------------------------

RT_PROGRAM void trace_ray()
{
	size_t2 screen = output_buffer.size();

	float2 d = make_float2(launch_index) / make_float2(screen) * 2.f - 1.f;
	float3 ray_origin = eye;
	float3 ray_direction = normalize(d.x*U + d.y*V + W);

	Ray ray(ray_origin, ray_direction, 0, EPSILON);

	PerRayData_radiance prd;
	prd.importance = 1.f;
	prd.depth = 0;

	rtTrace(scene_geometry, ray, prd);

	output_buffer[launch_index] = make_float4(prd.result, 1.0f);
}

//-----------------------------------------------------------------------------
// Lambertian surface closest-hit
//-----------------------------------------------------------------------------

rtDeclareVariable(float3, Ka, , );
rtDeclareVariable(float3, Ks, , );
rtDeclareVariable(float, phong_exp, , );
rtDeclareVariable(float3, Kd, , );
rtDeclareVariable(float3, ambient_light_color, , );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );

struct BasicLight
{
#if defined(__cplusplus)
	typedef optix::float3 float3;
#endif
	float3 pos;
	float3 color;
	int    casts_shadow;
	int    padding;
};

rtBuffer<BasicLight> lights;


RT_PROGRAM void diffuse()
{
	float3 world_geo_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
	float3 world_shade_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	float3 ffnormal = faceforward(world_shade_normal, -ray.direction, world_geo_normal);
	float3 color = Ka * ambient_light_color;

	float3 hit_point = ray.origin + t_hit * ray.direction;

	for(int i = 0; i < lights.size(); ++i) {
		BasicLight light = lights[i];
		float3 L = normalize(light.pos - hit_point);
		float nDl = dot(ffnormal, L);

		if(nDl > 0.0f) {
			// cast shadow ray
			PerRayData_shadow shadow_prd;
			shadow_prd.attenuation = make_float3(1.0f);
			float Ldist = length(light.pos - hit_point);
			Ray shadow_ray(hit_point, L, 1, EPSILON, Ldist);
			rtTrace(scene_geometry, shadow_ray, shadow_prd);
			float3 light_attenuation = shadow_prd.attenuation;

			if(fmaxf(light_attenuation) > 0.0f) {
				float3 Lc = light.color * light_attenuation;
				color += Kd * nDl * Lc;

				float3 H = normalize(L - ray.direction);
				float nDh = dot(ffnormal, H);
				if(nDh > 0)
					color += Ks * Lc * pow(nDh, phong_exp);
			}

		}
	}
	prd_radiance.result = color;
}

//-----------------------------------------------------------------------------
// Shadow any-hit
//-----------------------------------------------------------------------------

RT_PROGRAM void shadow()
{
	prd_shadow.attenuation = make_float3(0);
	rtTerminateRay();
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