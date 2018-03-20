#include <optixu/optixu_math_namespace.h>

using namespace optix;

//--------------------------------------------------------------
// Calculates the disparity between the two input buffers
// This program assumes that the input buffers
// are in the range [0, 1]
//--------------------------------------------------------------

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtBuffer<float3, 2> input_buffer_0;
rtBuffer<float3, 2> input_buffer_1;
rtBuffer<float3, 2> difference_buffer;

RT_PROGRAM void calculate_difference()
{
	const float3 diff = input_buffer_0[launch_index] - input_buffer_1[launch_index];
	difference_buffer[launch_index] = make_float3(dot(diff, diff) / 3.f * 20.f);
}
