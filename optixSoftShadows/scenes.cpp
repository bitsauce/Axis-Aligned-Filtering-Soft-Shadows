#include "scenes.h"
#include "geometry.h"

//--------------------------------------------------------------
// Default Scene
//--------------------------------------------------------------

DefaultScene::DefaultScene()
{
	// Setup light
	light.corner = make_float3(343.0f, 520.0f, 227.0f);
	light.v1 = make_float3(-130.0f, 0.0f, 0.0f);
	light.v2 = make_float3(0.0f, 0.0f, 130.0f);
	light.normal = normalize(cross(light.v1, light.v2));
	light.emission = make_float3(15.0f, 15.0f, 5.0f);

	lightBuffer = context->createBuffer(RT_BUFFER_INPUT);
	lightBuffer->setFormat(RT_FORMAT_USER);
	lightBuffer->setElementSize(sizeof(ParallelogramLight));
	lightBuffer->setSize(1u);
	memcpy(lightBuffer->map(), &light, sizeof(light));
	lightBuffer->unmap();
	context["lights"]->setBuffer(lightBuffer);

	// Material
	Material diffuse = context->createMaterial();
	diffuse->setClosestHitProgram(GROUND_TRUTH_RAY, context->createProgramFromPTXString(cudaFiles["ground_truth"], "diffuse"));
	diffuse->setClosestHitProgram(GEOMETRY_HIT_RAY, context->createProgramFromPTXString(cudaFiles["main"], "sample_geometry_hit"));
	diffuse->setAnyHitProgram(SHADOW_RAY, context->createProgramFromPTXString(cudaFiles["main"], "shadow"));

	diffuse["Ka"]->setFloat(0.3f, 0.3f, 0.3f);
	diffuse["Kd"]->setFloat(0.6f, 0.7f, 0.8f);
	diffuse["Ks"]->setFloat(0.8f, 0.9f, 0.8f);
	diffuse["phong_exp"]->setFloat(88);
	diffuse["reflectivity_n"]->setFloat(0.2f, 0.2f, 0.2f);

	// create geometry instances
	std::vector<GeometryInstance> gis;

	const float3 white = make_float3(0.8f, 0.8f, 0.8f);
	const float3 green = make_float3(0.05f, 0.8f, 0.05f);
	const float3 red = make_float3(0.8f, 0.05f, 0.05f);
	const float3 light_em = make_float3(15.0f, 15.0f, 5.0f);

	// Floor
	gis.push_back(createParallelogram(make_float3(0.0f, 0.0f, 0.0f),
									  make_float3(0.0f, 0.0f, 560.0f),
									  make_float3(560.0f, 0.0f, 0.0f),
									  diffuse,
									  white));

	// Ceiling
	gis.push_back(createParallelogram(make_float3(0.0f, 560.0f, 0.0f),
									  make_float3(560.0f, 0.0f, 0.0f),
									  make_float3(0.0f, 0.0f, 560.0f),
									  diffuse,
									  white));

	// Back wall
	gis.push_back(createParallelogram(make_float3(0.0f, 0.0f, 560.0f),
									  make_float3(0.0f, 560.0f, 0.0f),
									  make_float3(560.0f, 0.0f, 0.0f),
									  diffuse,
									  white));

	// Right wall
	gis.push_back(createParallelogram(make_float3(0.0f, 0.0f, 0.0f),
									  make_float3(0.0f, 560.0f, 0.0f),
									  make_float3(0.0f, 0.0f, 560.0f),
									  diffuse,
									  green));

	// Left wall
	gis.push_back(createParallelogram(make_float3(560.0f, 0.0f, 0.0f),
									  make_float3(0.0f, 0.0f, 560.0f),
									  make_float3(0.0f, 560.0f, 0.0f),
									  diffuse,
									  red));

	// Short block
	gis.push_back(createParallelogram(make_float3(130.0f, 165.0f, 65.0f),
									  make_float3(-48.0f, 0.0f, 160.0f),
									  make_float3(160.0f, 0.0f, 49.0f),
									  diffuse,
									  white));
	gis.push_back(createParallelogram(make_float3(290.0f, 0.0f, 114.0f),
									  make_float3(0.0f, 165.0f, 0.0f),
									  make_float3(-50.0f, 0.0f, 158.0f),
									  diffuse,
									  white));
	gis.push_back(createParallelogram(make_float3(130.0f, 0.0f, 65.0f),
									  make_float3(0.0f, 165.0f, 0.0f),
									  make_float3(160.0f, 0.0f, 49.0f),
									  diffuse,
									  white));
	gis.push_back(createParallelogram(make_float3(82.0f, 0.0f, 225.0f),
									  make_float3(0.0f, 165.0f, 0.0f),
									  make_float3(48.0f, 0.0f, -160.0f),
									  diffuse,
									  white));
	gis.push_back(createParallelogram(make_float3(240.0f, 0.0f, 272.0f),
									  make_float3(0.0f, 165.0f, 0.0f),
									  make_float3(-158.0f, 0.0f, -47.0f),
									  diffuse,
									  white));

	// Tall block
	gis.push_back(createParallelogram(make_float3(423.0f, 330.0f, 247.0f),
									  make_float3(-158.0f, 0.0f, 49.0f),
									  make_float3(49.0f, 0.0f, 159.0f),
									  diffuse,
									  white));
	gis.push_back(createParallelogram(make_float3(423.0f, 0.0f, 247.0f),
									  make_float3(0.0f, 330.0f, 0.0f),
									  make_float3(49.0f, 0.0f, 159.0f),
									  diffuse,
									  white));
	gis.push_back(createParallelogram(make_float3(472.0f, 0.0f, 406.0f),
									  make_float3(0.0f, 330.0f, 0.0f),
									  make_float3(-158.0f, 0.0f, 50.0f),
									  diffuse,
									  white));
	gis.push_back(createParallelogram(make_float3(314.0f, 0.0f, 456.0f),
									  make_float3(0.0f, 330.0f, 0.0f),
									  make_float3(-49.0f, 0.0f, -160.0f),
									  diffuse,
									  white));
	gis.push_back(createParallelogram(make_float3(265.0f, 0.0f, 296.0f),
									  make_float3(0.0f, 330.0f, 0.0f),
									  make_float3(158.0f, 0.0f, -49.0f),
									  diffuse,
									  white));

	// Create geometry group
	GeometryGroup geometry_group = context->createGeometryGroup(gis.begin(), gis.end());
	geometry_group->setAcceleration(context->createAcceleration("NoAccel"));
	context["scene_geometry"]->set(geometry_group);
}

void DefaultScene::update()
{
	if(animate)
	{
		light.corner = make_float3(343.0f + cos(glutGet(GLUT_ELAPSED_TIME) / 1000.f) * 100.f,
								   520.0f,
								   227.0f + sin(glutGet(GLUT_ELAPSED_TIME) / 1000.f) * 100.f);
		memcpy(lightBuffer->map(), &light, sizeof(light));
		lightBuffer->unmap();
		context["lights"]->setBuffer(lightBuffer);
	}
}

//--------------------------------------------------------------
// Grid Scene
//--------------------------------------------------------------

GridScene::GridScene()
{
	// Setup light
	light.corner = make_float3(400.0f, 520.0f, 500.0f);
	light.v1 = make_float3(-130.0f, 0.0f, 0.0f);
	light.v2 = make_float3(0.0f, 0.0f, 130.0f);
	light.normal = normalize(cross(light.v1, light.v2));
	light.emission = make_float3(15.0f, 15.0f, 5.0f);

	lightBuffer = context->createBuffer(RT_BUFFER_INPUT);
	lightBuffer->setFormat(RT_FORMAT_USER);
	lightBuffer->setElementSize(sizeof(ParallelogramLight));
	lightBuffer->setSize(1u);
	memcpy(lightBuffer->map(), &light, sizeof(light));
	lightBuffer->unmap();
	context["lights"]->setBuffer(lightBuffer);

	// Material
	Material diffuse = context->createMaterial();
	diffuse->setClosestHitProgram(GROUND_TRUTH_RAY, context->createProgramFromPTXString(cudaFiles["ground_truth"], "diffuse"));
	diffuse->setClosestHitProgram(GEOMETRY_HIT_RAY, context->createProgramFromPTXString(cudaFiles["main"], "sample_geometry_hit"));
	diffuse->setAnyHitProgram(SHADOW_RAY, context->createProgramFromPTXString(cudaFiles["main"], "shadow"));

	diffuse["Ka"]->setFloat(0.3f, 0.3f, 0.3f);
	diffuse["Kd"]->setFloat(0.6f, 0.7f, 0.8f);
	diffuse["Ks"]->setFloat(0.8f, 0.9f, 0.8f);
	diffuse["phong_exp"]->setFloat(88);
	diffuse["reflectivity_n"]->setFloat(0.2f, 0.2f, 0.2f);

	// create geometry instances
	std::vector<GeometryInstance> gis;

	const float3 white = make_float3(0.8f, 0.8f, 0.8f);
	const float3 green = make_float3(0.05f, 0.8f, 0.05f);
	const float3 red = make_float3(0.8f, 0.05f, 0.05f);
	const float3 blue = make_float3(0.5f, 0.5f, 0.8f);

	// Floor
	gis.push_back(createParallelogram(make_float3(0.0f, 0.0f, 0.0f),
									  make_float3(0.0f, 0.0f, 1000.0f),
									  make_float3(1000.0f, 0.0f, 0.0f),
									  diffuse,
									  make_float3(0.8f, 0.8f, 0.5f)));

	// Load mesh
	Matrix4x4 matrix;
	matrix = Matrix4x4::translate(make_float3(500, 80, 500)) * Matrix4x4::scale(make_float3(20, 20, 20)) * Matrix4x4::rotate(35, make_float3(1.0f, 0.0f, 0.0f)) * Matrix4x4::rotate(15, make_float3(0.0f, 1.0f, 0.0f));
	gis.push_back(loadMesh("meshes/grid.obj", diffuse, blue, matrix));
	matrix = Matrix4x4::translate(make_float3(300, 0, 300)) * Matrix4x4::scale(make_float3(80, 80, 80));
	gis.push_back(loadMesh("meshes/daisy2.obj", diffuse, green, matrix));

	// Create geometry group
	GeometryGroup geometry_group = context->createGeometryGroup(gis.begin(), gis.end());
	geometry_group->setAcceleration(context->createAcceleration("Trbvh"/*"NoAccel"*/));
	context["scene_geometry"]->set(geometry_group);
}

void GridScene::update()
{
	/*if(animate)
	{
		light.corner = make_float3(343.0f + cos(glutGet(GLUT_ELAPSED_TIME) / 1000.f) * 100.f,
								   520.0f,
								   227.0f + sin(glutGet(GLUT_ELAPSED_TIME) / 1000.f) * 100.f);
		memcpy(lightBuffer->map(), &light, sizeof(light));
		lightBuffer->unmap();
		context["lights"]->setBuffer(lightBuffer);
	}*/
}
