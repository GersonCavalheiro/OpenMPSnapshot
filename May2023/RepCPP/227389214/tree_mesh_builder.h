

#ifndef TREE_MESH_BUILDER_H
#define TREE_MESH_BUILDER_H


#include "base_mesh_builder.h"

class TreeMeshBuilder : public BaseMeshBuilder {
public:
TreeMeshBuilder(unsigned gridEdgeSize);

protected:
unsigned octree(const ParametricScalarField &field, unsigned mGridSize, const Vec3_t<float> &pos);

unsigned marchCubes(const ParametricScalarField &field);

float evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field);

void emitTriangle(const Triangle_t &triangle);

const Triangle_t *getTrianglesArray() const { return mTriangles.data(); }

std::vector <Triangle_t> mTriangles; 

};

#endif 
