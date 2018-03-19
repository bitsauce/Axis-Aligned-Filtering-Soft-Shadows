#pragma once

#include "common.h"

GeometryInstance createParallelogram(const float3& anchor, const float3& offset1, const float3& offset2, Material material, const float3& color);
GeometryInstance loadMesh(const std::string& filename, Material material, const float3& color, const Matrix4x4 &transformationMatrix = Matrix4x4::identity());
