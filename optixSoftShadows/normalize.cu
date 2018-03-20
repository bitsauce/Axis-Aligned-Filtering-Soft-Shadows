#include <optixu/optixu_math_namespace.h>

using namespace optix;

//--------------------------------------------------------------
// Normalizes the input buffer by dividing its components by
// the max value
//--------------------------------------------------------------

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtBuffer<float, 2> normalize_buffer;
rtBuffer<float3, 2> heatmap_buffer;
rtDeclareVariable(float, max_value, , );

RT_PROGRAM void normalize()
{
	normalize_buffer[launch_index] /= max_value;

	float greyValue = normalize_buffer[launch_index];
	float3 heat = make_float3(0.f, 0.f, 0.f);

	heat.x = smoothstep(0.5f, 0.8f, greyValue);
	if (greyValue >= 0.90f) {
		heat.x *= (1.1f - greyValue) * 5.0f;
	}
	if (greyValue > 0.7f) {
		heat.y = smoothstep(1.0f, 0.7f, greyValue);
	}
	else {
		heat.y = smoothstep(0.0f, 0.7f, greyValue);
	}
	heat.z = smoothstep(1.0f, 0.0f, greyValue);
	if (greyValue <= 0.3f) {
		heat.z *= greyValue / 0.3f;
	}

	heatmap_buffer[launch_index] = heat;
}
