#include <optixu/optixu_math_namespace.h>

using namespace optix;

//--------------------------------------------------------------
// Gaussian blur - using input_buffer.w as filter size
//--------------------------------------------------------------

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtBuffer<float4, 2> diffuse_buffer;
rtBuffer<float,  2> beta_buffer;
rtBuffer<float,  2> object_id_buffer;
rtBuffer<float2, 2> projected_distances_buffer;
rtBuffer<float4, 2> blur_h_buffer;
rtBuffer<float4, 2> blur_v_buffer;

float gauss1D(const float x, const float std)
{
	const float sqrt_2_pi = sqrtf(2.f * M_PIf);
	return expf(-(x * x) / (2.f * std * std)) / (sqrt_2_pi * std);
}

RT_PROGRAM void blurH()
{
	size_t2 screen = diffuse_buffer.size();
	const float beta = beta_buffer[launch_index];
	const int kernel_size = 25.f * beta / 16.0f; // TODO: Experiment with different kernel_sizes -- kernel as a function of beta?

	if(beta == 0.f) {
		blur_h_buffer[launch_index] = make_float4(make_float3(diffuse_buffer[launch_index]), 1.f);
		return;
	}

	uint object_id = uint(object_id_buffer[launch_index]);
	float3 color = make_float3(0.f);
	float sum = 0.f;
	for(int i = -kernel_size; i <= kernel_size; i++)
	{
		// Explointing interger underflow for when pos.x < 0
		const uint2 pos = make_uint2(launch_index.x + i, launch_index.y);
		if(pos.x >= screen.x || object_id != object_id_buffer[pos]) continue;

		float2 center = projected_distances_buffer[launch_index];
		float2 p = projected_distances_buffer[pos];
		const float offset = length(center - p);

		const float w = gauss1D(offset/300.0f, beta);
		color += make_float3(diffuse_buffer[pos]) * w;
		sum += w;
	}

	blur_h_buffer[launch_index] = make_float4(color / sum, 1.f);
}

RT_PROGRAM void blurV()
{
	size_t2 screen = diffuse_buffer.size();
	const float beta = beta_buffer[launch_index];
	const int kernel_size = 25.f * beta / 16.0f;

	if(beta == 0.f) {
		blur_v_buffer[launch_index] = make_float4(make_float3(diffuse_buffer[launch_index]), 1.f);
		return;
	}

	uint object_id = uint(object_id_buffer[launch_index]);
	float3 color = make_float3(0.f);
	float sum = 0.f;
	for(int i = -kernel_size; i <= kernel_size; i++)
	{
		const uint2 pos = make_uint2(launch_index.x, launch_index.y + i);
		if(pos.y >= screen.y || object_id != object_id_buffer[pos]) continue;

		float2 center = projected_distances_buffer[launch_index];
		float2 p = projected_distances_buffer[pos];
		const float offset = length(center - p);

		const float w = gauss1D(offset / 300.0f, beta);
		color += make_float3(blur_h_buffer[pos]) * w;
		sum += w;
	}

	blur_v_buffer[launch_index] = make_float4(color / sum, 1.f);
}