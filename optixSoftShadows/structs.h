#pragma once

#include <optixu/optixu_math_namespace.h> 

using namespace optix;

enum RayTypes
{
	DIFFUSE_RAY,
	SHADOW_RAY
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