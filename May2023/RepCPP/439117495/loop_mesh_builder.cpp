

#include <iostream>
#include <math.h>
#include <limits>

#include "loop_mesh_builder.h"

LoopMeshBuilder::LoopMeshBuilder(unsigned gridEdgeSize)
: BaseMeshBuilder(gridEdgeSize, "OpenMP Loop")
{

}

unsigned LoopMeshBuilder::marchCubes(const ParametricScalarField &field)
{
size_t totalCubesCount = mGridSize*mGridSize*mGridSize;

unsigned totalTriangles = 0;

#pragma omp parallel for \
default(none) \
schedule(static, 32) \
shared(totalCubesCount, field) \
reduction(+: totalTriangles)
for(size_t i = 0; i < totalCubesCount; ++i)
{
Vec3_t<float> cubeOffset( i % mGridSize,
(i / mGridSize) % mGridSize,
i / (mGridSize*mGridSize));

totalTriangles += buildCube(cubeOffset, field);
}

return totalTriangles;
}

float LoopMeshBuilder::evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field)
{

const Vec3_t<float> *pPoints = field.getPoints().data();
const unsigned count = unsigned(field.getPoints().size());

float value = std::numeric_limits<float>::max();

for(unsigned i = 0; i < count; ++i)
{
float distanceSquared  = (pos.x - pPoints[i].x) * (pos.x - pPoints[i].x);
distanceSquared       += (pos.y - pPoints[i].y) * (pos.y - pPoints[i].y);
distanceSquared       += (pos.z - pPoints[i].z) * (pos.z - pPoints[i].z);

value = std::min(value, distanceSquared);
}

return sqrt(value);
}

void LoopMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t &triangle)
{

#pragma omp critical(crit)
mTriangles.push_back(triangle);
}
