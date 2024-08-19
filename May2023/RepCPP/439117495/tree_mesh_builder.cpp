

#include <iostream>
#include <math.h>
#include <limits>

#include "tree_mesh_builder.h"

TreeMeshBuilder::TreeMeshBuilder(unsigned gridEdgeSize)
: BaseMeshBuilder(gridEdgeSize, "Octree")
{

}

bool TreeMeshBuilder::checkMaxDepth(unsigned mGridSize) {
return (mGridSize > CUTOFF) ? false : true;
}

bool TreeMeshBuilder::checkCondition(const ParametricScalarField field, const Vec3_t<float> cubeCenter, unsigned mGridSize) {
float evalField = evaluateFieldAt(cubeCenter, field);
float condition = mIsoLevel + ((sqrtf(3.0f) / 2.0f) * mGridSize * mGridResolution);

return (evalField > condition) ? false : true;
}

unsigned TreeMeshBuilder::generateOctree(const ParametricScalarField &field, unsigned mGridSize, const Vec3_t<float> &position) {
size_t totalCubesCount = mGridSize * mGridSize * mGridSize;
float mGridSizeMid = mGridSize / 2.0f;
unsigned totalTriangles = 0;
Vec3_t<float> cubeCenter(
(position.x + mGridSizeMid) * mGridResolution,
(position.y + mGridSizeMid) * mGridResolution,
(position.z + mGridSizeMid) * mGridResolution
);

if (checkCondition(field, cubeCenter, mGridSize)) {
totalTriangles = 0;

if (checkMaxDepth(mGridSize)) {
return buildCube(position, field);
}



for (size_t i = 0; i < OCTREE; i++) {

Vec3_t<float> nextPosition(
position.x + (sc_vertexNormPos[i].x * mGridSizeMid),
position.y + (sc_vertexNormPos[i].y * mGridSizeMid),
position.z + (sc_vertexNormPos[i].z * mGridSizeMid)
);

#pragma omp task shared(field, totalTriangles, mGridSizeMid)
#pragma omp atomic update
totalTriangles += generateOctree(field, mGridSizeMid, nextPosition);
}

#pragma omp taskwait
return totalTriangles;

} else {
return 0.0f;
}
}

unsigned TreeMeshBuilder::marchCubes(const ParametricScalarField &field)
{

size_t totalCubesCount = mGridSize * mGridSize * mGridSize;

unsigned totalTriangles = 0;
float start = 0.0f;
Vec3_t<float> initPosition(start, start, start);

#pragma omp parallel shared(field, totalTriangles)
#pragma omp single nowait
totalTriangles = generateOctree(field, mGridSize, initPosition);

return totalTriangles;
}

float TreeMeshBuilder::evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field)
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

void TreeMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t &triangle)
{

#pragma omp critical(critical)
mTriangles.push_back(triangle);
}
