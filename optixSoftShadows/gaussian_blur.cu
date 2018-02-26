#include <optixu/optixu_math_namespace.h>

using namespace optix;

//--------------------------------------------------------------
// Gaussian blur - using input_buffer.w as filter size
//--------------------------------------------------------------

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtBuffer<float4, 2> main_output;
rtBuffer<float4, 2> blur_output;

float gauss(const float x, const float y, const float std)
{
	return expf(-(x * x + y * y) / (2.f * std * std)) / (2.f * M_PIf * std * std);
}

RT_PROGRAM void blur()
{
	size_t2 screen = main_output.size();
	const int kernel_size = int(main_output[launch_index].w) / 2;

	float3 color = make_float3(0.f);
	float weight = 0.f;
	uint2 pixel_index;
	for(int y = -kernel_size; y <= kernel_size; y++)
	{
		pixel_index.y = launch_index.y + y; // Expoiting interger underflow
		if(pixel_index.y >= screen.y) continue;
		for(int x = -kernel_size; x <= kernel_size; x++)
		{
			pixel_index.x = launch_index.x + x;
			if(pixel_index.x >= screen.x) continue;

			const float g = gauss(x, y, 1.f);
			color += make_float3(main_output[pixel_index]) * g;
			weight += g;
		}
	}

	blur_output[launch_index] = make_float4(color / weight, 1.f);
}
