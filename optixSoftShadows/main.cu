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

#define EPSILON  1.e-3f

//--------------------------------------------------------------
// Per-ray data structs
//--------------------------------------------------------------

struct PerRayData_diffuse
{
	float3       color;         // Diffuse color
	float        depth;			// Sample depth
	float2       projected_distance;
	float        object_id;
	float        beta;			// Filter width (screen-space standard deviation)
	float		 num_samples;	// Number of adaptive samples
	unsigned int seed;          // Seed for random sampling
};

struct PerRayData_shadow
{
	float3 attenuation;
	float3 hit_point;
};

//--------------------------------------------------------------
// Variable declarations
//--------------------------------------------------------------

// Input pixel-coordinate
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );

rtBuffer<float4, 2> diffuse_buffer;             // Diffuse color buffer
rtBuffer<float,  2> beta_buffer;                // Beta buffer (gaussian standard deviation)
rtBuffer<float,  2> depth_buffer;               // Depth buffer
rtBuffer<float,  2> object_id_buffer;           // Object id buffer
rtBuffer<float,  2> num_samples_buffer;           // Sample number buffer
rtBuffer<float2, 2> projected_distances_buffer; // Projected distances buffer (offset of screen-space gaussian)

// Scene geometry objects
rtDeclareVariable(rtObject, scene_geometry, , );

// Pinhole camera variables
rtDeclareVariable(float3, eye, , );
rtDeclareVariable(float3, U,   , );
rtDeclareVariable(float3, V,   , );
rtDeclareVariable(float3, W,   , );

rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );

rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(PerRayData_diffuse, prd_diffuse, rtPayload, );
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );

// Light sources
rtBuffer<ParallelogramLight> lights;

//--------------------------------------------------------------
// Main ray program
//--------------------------------------------------------------

RT_PROGRAM void trace_ray()
{
	size_t2 screen = diffuse_buffer.size(); // Screen size
	float2 d = make_float2(launch_index) / make_float2(screen) * 2.f - 1.f; // Pixel coordinate in [-1, 1]
	float3 ray_origin = eye;
	float3 ray_direction = normalize(d.x*U + d.y*V + W);

	// Create ray from camera into scene
	Ray ray(ray_origin, ray_direction, DIFFUSE_RAY, EPSILON);

	// Per radiance data
	PerRayData_diffuse prd;
	prd.seed = tea<16>(screen.x*launch_index.y + launch_index.x, 0);//frame_number);
	prd.beta = 0.f;

	// Trace geometry
	rtTrace(scene_geometry, ray, prd);

	// Set resulting diffuse color and beta
	diffuse_buffer[launch_index] = make_float4(prd.color, 1.f);
	beta_buffer[launch_index] = prd.beta;
	depth_buffer[launch_index] = prd.depth;
	projected_distances_buffer[launch_index] = prd.projected_distance;
	object_id_buffer[launch_index] = prd.object_id;
	num_samples_buffer[launch_index] = prd.num_samples;
}

//-----------------------------------------------------------------------------
// Lambertian surface closest-hit
//-----------------------------------------------------------------------------

rtDeclareVariable(float3, diffuse_color, , );
rtDeclareVariable(float3, Ka, , );
rtDeclareVariable(float3, Ks, , );
rtDeclareVariable(float, phong_exp, , );
rtDeclareVariable(float3, Kd, , );
rtDeclareVariable(float3, ambient_light_color, , );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(uint, object_id, , );

#define FLT_MAX 3.402823466e+38F

RT_PROGRAM void sample_distances(unsigned int& seed, ParallelogramLight light, float3 ffnormal, float3 hit_point, float& d2_min, float& d2_max)
{
	// Choose random point on light
	const float z1 = rnd(seed);
	const float z2 = rnd(seed);
	const float3 light_pos = light.corner + light.v1 * z1 + light.v2 * z2;

	float3 L = normalize(light_pos - hit_point);
	float nDl = dot(ffnormal, L);
	//if(nDl > 0.0f) // Check if light is behind
	{
		// TODO: Maybe d1 should be average of these?
		float Ldist = length(light_pos - hit_point);

		// Cast shadow ray
		PerRayData_shadow shadow_prd;
		shadow_prd.attenuation = make_float3(1.0f);

		Ray shadow_ray(hit_point, L, SHADOW_RAY, EPSILON, Ldist);
		rtTrace(scene_geometry, shadow_ray, shadow_prd);

		float3 light_attenuation = shadow_prd.attenuation;
		if (fmaxf(light_attenuation) <= 0.0f) // If light source was occluded
		{
			const float d2 = length(shadow_prd.hit_point - light_pos);

			// Store min d2
			if (d2 < d2_min)
			{
				d2_min = d2;
			}

			// Store max d2
			if (d2 > d2_max)
			{
				d2_max = d2;
			}
		}
	}
}

RT_PROGRAM void diffuse()
{
	float3 world_geo_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
	float3 world_shade_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	float3 ffnormal = faceforward(world_shade_normal, -ray.direction, world_geo_normal);
	float3 color = Ka * ambient_light_color;

	float3 hit_point = ray.origin + t_hit * ray.direction;
	
	unsigned int seed = prd_diffuse.seed;
	for(int i = 0; i < lights.size(); ++i)
	{
		ParallelogramLight light = lights[i];
		const float3 light_center = light.corner + light.v1 * 0.5f + light.v2 * 0.5f;

		Matrix3x3 projection_matrix;
		projection_matrix.setRow(0, light.v1);
		projection_matrix.setRow(1, light.v2);
		projection_matrix.setRow(2, light.normal);

		float3 p_projected = projection_matrix * hit_point;
		prd_diffuse.projected_distance = make_float2(p_projected);


		// Sample color
		float3 L = normalize(light_center - hit_point);
		float nDl = dot(ffnormal, L);
		if (nDl > 0.0f) // Check if light is behind
		{
			float Ldist = length(light_center - hit_point);

			// Cast shadow ray
			PerRayData_shadow shadow_prd;
			shadow_prd.attenuation = make_float3(1.0f);

			Ray shadow_ray(hit_point, L, SHADOW_RAY, EPSILON, Ldist);
			rtTrace(scene_geometry, shadow_ray, shadow_prd);

			float3 light_attenuation = shadow_prd.attenuation;
			if (fmaxf(light_attenuation) > 0.0f) // If we hit the light
			{
				float3 Lc = light_attenuation * diffuse_color;
				color += Kd * nDl * Lc;

				// Apply specularity
				float3 H = normalize(L - ray.direction);
				float nDh = dot(ffnormal, H);
				if (nDh > 0)
				{
					color += Ks * Lc * pow(nDh, phong_exp);
				}
			}
		}

		// Send 9 rays
		float d2_min = FLT_MAX;  // Min distance from light to occluder
		float d2_max = -FLT_MAX; // Max distance from light to occluder
		float d1 = length(hit_point - light_center); // Distance from light to receiver
		for(int j = 0; j < 9; j++)
		{
			sample_distances(seed, light, ffnormal, hit_point, d2_min, d2_max);
		}

		// DEBUG: Show the light
		if(d1 < 10.0f)
		{
			color = make_float3(1.f, 1.f, 1.f);
		}

		// Constants from the paper
		const float k = 3.f;
		const float alpha = 1.f;
		const float mu = 2.f;
		const float max_num_samples = 50.f;

		// Standard deviation of Gaussian of the light
		// TODO: Experiment with different sigmas
		const float sigma = 130.f / 2.f;

		const float s1 = (d1 / d2_min) - 1.f;
		const float s2 = (d1 / d2_max) - 1.f;
		const float inv_s2 = alpha / (1.f + s2);

		const float depth = length(t_hit * ray.direction);
		const float omega_max_pix = 1.f / depth;
		// TODO: d should be calculated as the average 3D eucledean distance between this pixels hitpoint and it's neighbouring pixels
		// persumably this should be calculated in its own pass beforehand
		const float omega_max_x = inv_s2 * omega_max_pix;

		// If this pixel was occluded (that is, d2_max > 0)
		if(d2_max > 0.f)
		{
			// Calculate filter width at current pixel
			const float beta = 1.f / k * 1.f / mu * max(sigma * ((d1 / d2_max) - 1.f), //1.f / omega_max_x); // TODO: Calculate the omega_max_x and use it
																					   -1000.0f);
			prd_diffuse.beta = beta;
		}

		// Calculate pixel area and light area
		const float Ap = 1.f / (omega_max_pix * omega_max_pix);
		const float Al = 4.f * sigma * sigma;

		// Calcuate number of additional samples
		const float num_samples = 50.0f; // min(mu * 2 / s2 * sqrtf(Ap / Al), max_num_samples); //min(4.f * powf(1.f + mu * (s1 / s2), 2.f) * powf(mu * 2 / s2 * sqrtf(Ap / Al) + inv_s2, 2.f), max_num_samples);
		prd_diffuse.num_samples = num_samples;

		for (int j = 0; j < (int)num_samples; j++)
		{
			sample_distances(seed, light, ffnormal, hit_point, d2_min, d2_max);
		}
	}
	prd_diffuse.color = color;
	prd_diffuse.depth = length(hit_point - ray.origin);
	prd_diffuse.object_id = float(object_id);
}

//-----------------------------------------------------------------------------
// Shadow any-hit
//-----------------------------------------------------------------------------

RT_PROGRAM void shadow()
{
	prd_shadow.hit_point = ray.origin + t_hit * ray.direction;
	prd_shadow.attenuation = make_float3(0);
	rtTerminateRay();
}

//--------------------------------------------------------------
// Miss program
//--------------------------------------------------------------

rtDeclareVariable(float3, bg_color,,);

RT_PROGRAM void miss()
{
	prd_diffuse.color = bg_color;
	prd_diffuse.depth = 0.f;
	prd_diffuse.object_id = 0.f;
	prd_diffuse.num_samples = 0.f;
}

//--------------------------------------------------------------
// Exception
//--------------------------------------------------------------

rtDeclareVariable(float3, bad_color,,);

RT_PROGRAM void exception()
{
	diffuse_buffer[launch_index] = make_float4(bad_color, 1.f);
	beta_buffer[launch_index] = 0.f;
	object_id_buffer[launch_index] = 0.f;
	num_samples_buffer[launch_index] = 0.f;
}