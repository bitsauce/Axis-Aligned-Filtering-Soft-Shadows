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

#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include "structs.h"
#include "random.h"

using namespace optix;

#define EPSILON  1.e-1f
#define FLT_MAX 3.402823466e+38F

//--------------------------------------------------------------
// Variable declarations
//--------------------------------------------------------------

// Input pixel-coordinate
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );

rtBuffer<float3, 2> diffuse_buffer;             // Diffuse color buffer
rtBuffer<float,  2> beta_buffer;                // Beta buffer (gaussian standard deviation)
rtBuffer<float,  2> d1_buffer;                  // Distance to light source
rtBuffer<float,  2> d2_min_buffer;              // Minimum distasnce to occluder
rtBuffer<float,  2> d2_max_buffer;              // Maximum distasnce to occluder
rtBuffer<float3, 2> geometry_hit_buffer;        // Geometry hit buffer
rtBuffer<float3, 2> geometry_normal_buffer;     // Geometry hit buffer
rtBuffer<float3, 2> ffnormal_buffer;            // For shading
rtBuffer<float,  2> object_id_buffer;           // Object id buffer
rtBuffer<float,  2> num_samples_buffer;         // Sample number buffer
rtBuffer<float2, 2> projected_distances_buffer; // Projected distances buffer (offset of screen-space gaussian)

// Scene geometry objects
rtDeclareVariable(rtObject, scene_geometry, , );

// Pinhole camera variables
rtDeclareVariable(float3, eye, , );
rtDeclareVariable(float3, U,   , );
rtDeclareVariable(float3, V,   , );
rtDeclareVariable(float3, W,   , );

rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(PerRayData_geometry_hit, prd_geometry_hit, rtPayload, );
rtDeclareVariable(PerRayData_distances, prd_distances, rtPayload, );
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );

// Light sources
rtBuffer<ParallelogramLight> lights;

// Geometry hit variables
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(float3, diffuse_color, , );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(uint, object_id, , );

//--------------------------------------------------------------
// Primary ray pass
//--------------------------------------------------------------

RT_PROGRAM void trace_primary_ray()
{
	size_t2 screen = diffuse_buffer.size(); // Screen size
	float2 d = make_float2(launch_index) / make_float2(screen) * 2.f - 1.f; // Pixel coordinate in [-1, 1]
	float3 ray_origin = eye;
	float3 ray_direction = normalize(d.x*U + d.y*V + W);

	// Create ray from camera into scene
	Ray ray(ray_origin, ray_direction, GEOMETRY_HIT_RAY, EPSILON);

	// Per radiance data
	PerRayData_geometry_hit prd;
	prd.color = make_float3(0.f);
	prd.object_id = 0;
	prd.geometry_hit = make_float3(0.f);
	prd.geometry_normal = make_float3(0.f);
	prd.ffnormal = make_float3(0.f);

	// Trace geometry
	rtTrace(scene_geometry, ray, prd);

	// Set resulting geometry hit coordinate
	diffuse_buffer[launch_index] = prd.color;
	object_id_buffer[launch_index] = prd.object_id;
	geometry_hit_buffer[launch_index] = prd.geometry_hit;
	geometry_normal_buffer[launch_index] = prd.geometry_normal;
	ffnormal_buffer[launch_index] = prd.ffnormal;
}

RT_PROGRAM void sample_geometry_hit()
{
	float3 world_geo_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
	float3 world_shade_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	float3 ffnormal = faceforward(world_shade_normal, -ray.direction, world_geo_normal);
	float3 hit_point = ray.origin + t_hit * ray.direction;

	prd_geometry_hit.color = diffuse_color;
	prd_geometry_hit.object_id = float(object_id);
	prd_geometry_hit.geometry_hit = hit_point;
	prd_geometry_hit.geometry_normal = world_geo_normal;
	prd_geometry_hit.ffnormal = ffnormal;
}

//--------------------------------------------------------------
// Distance sampling + adaptive sampling
//--------------------------------------------------------------

RT_PROGRAM void sample_distances_to_light(unsigned int& seed, float3 &color, ParallelogramLight light,
										  float3 ffnormal, float3 hit_point, float& d2_min, float& d2_max)
{
	// Choose random point on light
	const float z1 = rnd(seed);
	const float z2 = rnd(seed);
	const float3 light_pos = light.corner + light.v1 * z1 + light.v2 * z2;

	float3 L = normalize(light_pos - hit_point);
	float nDl = dot(ffnormal, L);
	if(nDl > 0.0f) // Check if light is behind
	{
		float Ldist = length(light_pos - hit_point);

		// Cast shadow ray
		PerRayData_shadow shadow_prd;
		shadow_prd.hit = false;

		Ray shadow_ray(hit_point, L, SHADOW_RAY, EPSILON, Ldist);
		rtTrace(scene_geometry, shadow_ray, shadow_prd);

		// If light source was occluded
		if(shadow_prd.hit)
		{
			const float d2 = length(shadow_prd.hit_point - light_pos);

			// Store min d2
			if(d2 < d2_min)
			{
				d2_min = d2;
			}

			// Store max d2
			if(d2 > d2_max)
			{
				d2_max = d2;
			}
		}
		else
		{
			const float3 Kd = make_float3(0.6f, 0.7f, 0.8f);
			color += Kd * nDl * diffuse_buffer[launch_index];
		}
	}
}

// Constants from the paper
const float k = 3.f;
const float alpha = 1.f;
const float mu = 2.f;
const float max_num_samples = 100.f;

// Standard deviation of Gaussian of the light
const float sigma = 130.f / 2.f;

RT_PROGRAM void sample_distances()
{
	size_t2 screen = geometry_hit_buffer.size();
	float3 ffnormal = ffnormal_buffer[launch_index];
	float3 hit_point = geometry_hit_buffer[launch_index];

	// Calculate projected distance per pixel
	float d = 0.f;
	if(launch_index.x > 0)        d += length(geometry_hit_buffer[make_uint2(launch_index.x - 1, launch_index.y)] - hit_point);
	if(launch_index.y > 0)        d += length(geometry_hit_buffer[make_uint2(launch_index.x, launch_index.y - 1)] - hit_point);
	if(launch_index.x < screen.x) d += length(geometry_hit_buffer[make_uint2(launch_index.x + 1, launch_index.y)] - hit_point);
	if(launch_index.y < screen.y) d += length(geometry_hit_buffer[make_uint2(launch_index.x, launch_index.y + 1)] - hit_point);
	d /= 4.f;
	const float omega_max_pix = 1.f / d;

	float3 color = make_float3(0.0f);
	unsigned int seed = tea<16>(screen.x*launch_index.y + launch_index.x, 0/*frame_number*/);
	for(int i = 0; i < lights.size(); ++i)
	{
		ParallelogramLight light = lights[i];
		const float3 light_center = light.corner + light.v1 * 0.5f + light.v2 * 0.5f;

		// Calculate distances parallel to the light source
		// (used as a offset in the gaussian blur)
		Matrix3x3 projection_matrix;
		projection_matrix.setCol(0, normalize(light.v1));
		projection_matrix.setCol(1, normalize(light.v2));
		projection_matrix.setCol(2, light.normal);

		float3 p_projected = projection_matrix * hit_point;
		projected_distances_buffer[launch_index] = make_float2(p_projected);

		// Send 9 rays
		float d2_min = FLT_MAX;  // Min distance from light to occluder
		float d2_max = -FLT_MAX; // Max distance from light to occluder
		float d1 = length(hit_point - light_center); // Distance from light to receiver
		for(int j = 0; j < 9; j++)
		{
			sample_distances_to_light(seed, color, light, ffnormal, hit_point, d2_min, d2_max);
		}

		// If this pixel was occluded (that is, d2_max > 0)
		if(d2_max > 0.f)
		{
			const float s1 = max(d1 / d2_min, 1.f) - 1.f;
			float s2 = max(d1 / d2_max, 1.f) - 1.f;
			float inv_s2 = alpha / (1.f + s2);

			// Calculate pixel area and light area
			const float Ap = 1.f / (omega_max_pix * omega_max_pix);
			const float Al = 4.f * sigma * sigma;

			// Calcuate number of additional samples
			const float num_samples = min(4.f * powf(1.f + mu * (s1 / s2), 2.f) * powf(mu * 2 / s2 * sqrtf(Ap / Al) + inv_s2, 2.f), max_num_samples);
			num_samples_buffer[launch_index] = num_samples;

			for(int j = 0; j < (int)num_samples; j++)
			{
				sample_distances_to_light(seed, color, light, ffnormal, hit_point, d2_min, d2_max);
			}

			color /= 9.f + num_samples;
		}
		else
		{
			// Set values for unoccluded pixels
			num_samples_buffer[launch_index] = 0.f;
			color /= 9.f;
			d1 = d2_min = d2_max = 0.f;
		}

		// Set sampled distances
		d1_buffer[launch_index] = d1;
		d2_min_buffer[launch_index] = d2_min;
		d2_max_buffer[launch_index] = d2_max;
	}

	// Set sampled color
	diffuse_buffer[launch_index] = color;
}

//-----------------------------------------------------------------------------
// Calculate beta
//-----------------------------------------------------------------------------

RT_PROGRAM void calculate_beta()
{
	// Calculate projected distance per pixel
	size_t2 screen = geometry_hit_buffer.size();
	float3 hit_point = geometry_hit_buffer[launch_index];
	float d = 0.f;
	if(launch_index.x > 0)        d += length(geometry_hit_buffer[make_uint2(launch_index.x - 1, launch_index.y)] - hit_point);
	if(launch_index.y > 0)        d += length(geometry_hit_buffer[make_uint2(launch_index.x, launch_index.y - 1)] - hit_point);
	if(launch_index.x < screen.x) d += length(geometry_hit_buffer[make_uint2(launch_index.x + 1, launch_index.y)] - hit_point);
	if(launch_index.y < screen.y) d += length(geometry_hit_buffer[make_uint2(launch_index.x, launch_index.y + 1)] - hit_point);
	d /= 4.f;
	const float omega_max_pix = 1.f / d;

	// Get d1, d2_max from previous pass
	float d2_max = d2_max_buffer[launch_index];
	float d1 = d1_buffer[launch_index];

	// For unocculded pixel, take the average in a 5 pixel radius
	if(d2_max == 0.f)
	{
		float sum = 0.f;
		for(int i = -5; i <= 5; i++)
		{
			for(int j = -5; j <= 5; j++)
			{
				const uint2 pos = make_uint2(launch_index.x + j, launch_index.y + i);
				if(pos.x >= screen.x || pos.y >= screen.y) continue;
				d1 += d1_buffer[pos];
				d2_max += d2_max_buffer[pos];
				sum += 1.f;
			}
		}

		// Get average
		d1 /= sum;
		d2_max /= sum;

		// Write back (for debug visualization)
		//d1_buffer[launch_index] = d1;
		//d2_max_buffer[launch_index] = d2_max;
	}

	// Make sure we can calculate beta
	if(d2_max > 0.f)
	{
		// Update s2 and inv_s2
		const float s2 = max(d1 / d2_max, 1.f) - 1.f;
		const float inv_s2 = alpha / (1.f + s2);
		const float omega_max_x = inv_s2 * omega_max_pix;

		// Calculate filter width at current pixel
		const float beta = 1.f / k * 1.f / mu * max(sigma * s2, 1.f / omega_max_x);
		beta_buffer[launch_index] = min(beta, 10.f);
	}
	else
	{
		// Pixel still unoccluded
		beta_buffer[launch_index] = 0.f;
	}
}

//-----------------------------------------------------------------------------
// Shadow any-hit
//-----------------------------------------------------------------------------

RT_PROGRAM void shadow()
{
	prd_shadow.hit_point = ray.origin + t_hit * ray.direction;
	prd_shadow.hit = true;
	rtTerminateRay();
}

//--------------------------------------------------------------
// Miss program
//--------------------------------------------------------------

rtDeclareVariable(float3, bg_color,,);

RT_PROGRAM void distances_miss()
{
	prd_distances.color = bg_color;
	prd_distances.projected_distance = make_float2(0.f);
	prd_distances.d1 = prd_distances.d2_min = prd_distances.d2_max = 0.f;
}

//--------------------------------------------------------------
// Exception
//--------------------------------------------------------------

rtDeclareVariable(float3, bad_color,,);

RT_PROGRAM void exception()
{
	diffuse_buffer[launch_index] = bad_color;
	beta_buffer[launch_index] = 0.f;
	object_id_buffer[launch_index] = 0.f;
	num_samples_buffer[launch_index] = 0.f;
	geometry_normal_buffer[launch_index] = make_float3(0.f);
}