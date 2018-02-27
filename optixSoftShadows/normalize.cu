#include <optixu/optixu_math_namespace.h>

using namespace optix;

//--------------------------------------------------------------
// Normalizes the input buffer by dividing its components by
// the max value
//--------------------------------------------------------------

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtBuffer<float, 2> normalize_buffer;
rtDeclareVariable(float, max_value, , );

RT_PROGRAM void normalize()
{
	normalize_buffer[launch_index] /= max_value;
}
