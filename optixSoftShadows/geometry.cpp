#include "geometry.h"

uint objectID = 0;

GeometryInstance createParallelogram(const float3& anchor, const float3& offset1, const float3& offset2, Material material, const float3& color)
{
	Program pgram_bounding_box = context->createProgramFromPTXString(cudaFiles["parallelogram"], "bounds");
	Program pgram_intersection = context->createProgramFromPTXString(cudaFiles["parallelogram"], "intersect");

	Geometry parallelogram = context->createGeometry();
	parallelogram->setPrimitiveCount(1u);
	parallelogram->setIntersectionProgram(pgram_intersection);
	parallelogram->setBoundingBoxProgram(pgram_bounding_box);

	float3 normal = normalize(cross(offset1, offset2));
	float d = dot(normal, anchor);
	float4 plane = make_float4(normal, d);

	float3 v1 = offset1 / dot(offset1, offset1);
	float3 v2 = offset2 / dot(offset2, offset2);

	parallelogram["plane"]->setFloat(plane);
	parallelogram["anchor"]->setFloat(anchor);
	parallelogram["v1"]->setFloat(v1);
	parallelogram["v2"]->setFloat(v2);

	GeometryInstance gi = context->createGeometryInstance();
	gi->setGeometry(parallelogram);
	gi["object_id"]->setUint(++objectID);
	gi->addMaterial(material);
	gi["diffuse_color"]->setFloat(color);
	return gi;
}

GeometryInstance loadMesh(const std::string& filename, Material material, const float3& color, const Matrix4x4 &transformationMatrix)
{
	OptiXMesh mesh;
	mesh.context = context;
	mesh.material = material;
	loadMesh(filename, mesh, transformationMatrix);

	GeometryInstance gi = mesh.geom_instance;
	gi["object_id"]->setUint(++objectID);
	gi->addMaterial(material);
	gi["diffuse_color"]->setFloat(color);
	return mesh.geom_instance;
}