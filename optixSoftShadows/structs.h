#pragma once

#include <optixu/optixu_math_namespace.h> 

using namespace optix;

enum RayTypes
{
	DIFFUSE_RAY,
	SHADOW_RAY,
	GROUND_TRUTH_RAY,
	NUM_RAYS
};

// Point light
struct BasicLight
{
	float3 pos;
	float3 color;
	int    casts_shadow;
	int    padding;
};

// Parallelogram light
struct ParallelogramLight
{
	float3 corner;
	float3 v1, v2;
	float3 normal;
	float3 emission;
};

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
	unsigned int num_samples;	// Number of adaptive samples
	unsigned int seed;          // Seed for random sampling
};

struct PerRayData_ground_truth
{
	float3       color;         // Diffuse color
	unsigned int seed;          // Seed for random sampling
};

struct PerRayData_shadow
{
	bool hit;
	float3 hit_point;
};