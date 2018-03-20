#include <optixu/optixu_math_namespace.h>

using namespace optix;

//--------------------------------------------------------------
// Gaussian blur - using input_buffer.w as filter size
//--------------------------------------------------------------

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtBuffer<float3, 2> diffuse_buffer;
rtBuffer<float,  2> beta_buffer;
rtBuffer<float,  2> object_id_buffer;
rtBuffer<float2, 2> projected_distances_buffer;
rtBuffer<float3, 2> blur_h_buffer;
rtBuffer<float3, 2> blur_v_buffer;
rtBuffer<float3, 2> geometry_normal_buffer;

float gauss1D(const float x, const float std)
{
	const float sqrt_2_pi = sqrtf(2.f * M_PIf);
	return expf(-(x * x) / (2.f * std * std)) / (sqrt_2_pi * std);
}

RT_PROGRAM void blurH()
{
	size_t2 screen = diffuse_buffer.size();
	const float beta = beta_buffer[launch_index];

	// TODO: Experiment with different kernel_sizes -- kernel as a function of beta?
	const int kernel_size = min(beta * 4.0f, 10.0f);

	if(beta == 0.f) {
		blur_h_buffer[launch_index] = diffuse_buffer[launch_index];
		return;
	}

	float object_id = object_id_buffer[launch_index];
	float3 geometry_normal = geometry_normal_buffer[launch_index];
	float3 color = make_float3(0.f);
	float sum = 0.f;
	float2 center = projected_distances_buffer[launch_index];
	for(int i = -kernel_size; i <= kernel_size; i++)
	{
		// Explointing interger underflow when pos.x < 0
		const uint2 pos = make_uint2(launch_index.x + i, launch_index.y);
		if(pos.x >= screen.x || object_id != object_id_buffer[pos]) continue;

		float2 p = projected_distances_buffer[pos];
		const float offset = length(center - p);

		const float w = gauss1D(offset, beta) * dot(geometry_normal, geometry_normal_buffer[pos]);
		color += diffuse_buffer[pos] * w;
		sum += w;
	}

	blur_h_buffer[launch_index] = color / sum;
}

RT_PROGRAM void blurV()
{
	size_t2 screen = diffuse_buffer.size();
	const float beta = beta_buffer[launch_index];
	const int kernel_size = min(beta * 4.0f, 10.0f);

	if(beta == 0.f) {
		blur_v_buffer[launch_index] = diffuse_buffer[launch_index];
		return;
	}
	
	float object_id = object_id_buffer[launch_index];
	float3 geometry_normal = geometry_normal_buffer[launch_index];
	float3 color = make_float3(0.f);
	float sum = 0.f;
	float2 center = projected_distances_buffer[launch_index];
	for(int i = -kernel_size; i <= kernel_size; i++)
	{
		const uint2 pos = make_uint2(launch_index.x, launch_index.y + i);
		if(pos.y >= screen.y || object_id != object_id_buffer[pos]) continue;

		float2 p = projected_distances_buffer[pos];
		const float offset = length(center - p);

		const float w = gauss1D(offset, beta) * dot(geometry_normal, geometry_normal_buffer[pos]);
		color += blur_h_buffer[pos] * w;
		sum += w;
	}

	blur_v_buffer[launch_index] = color / sum;
}
