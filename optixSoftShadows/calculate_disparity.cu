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
rtBuffer<float3, 2> disparity_buffer;

RT_PROGRAM void calculate_disparity()
{
	disparity_buffer[launch_index] = make_float3(length(input_buffer_0[launch_index] - input_buffer_1[launch_index]) / 4.f);
}
