#include "quadmesh.h"


QuadMesh::QuadMesh(int maxHalfEdges, int maxVertices) {
twins.resize(maxHalfEdges);
verts.resize(maxHalfEdges);
edges.resize(maxHalfEdges);
vertexCoords.resize(maxVertices);
}


void QuadMesh::resetMesh() { vertexCoords.fill({0, 0, 0}); }


int QuadMesh::next(int h) { return h % 4 == 3 ? h - 3 : h + 1; }


int QuadMesh::prev(int h) { return h % 4 == 0 ? h + 3 : h - 1; }


int QuadMesh::face(int h) { return h / 4; }


void QuadMesh::subdivideCatmullClark(QuadMesh& mesh) {
recalculateSizes(mesh);

#pragma omp parallel
{
#pragma omp for nowait
for (int h = 0; h < numHalfEdges; ++h) {
edgeRefinement(mesh, h);
}
#pragma omp for
for (int h = 0; h < numHalfEdges; h += 4) {
QVector3D c;
for (int j = 0; j < 4; j++) {
int v = vert(h + j);
c += vertexCoords.at(v);
}
int i = numVerts + face(h);
mesh.vertexCoords[i] = c / 4.0f;
}

#pragma omp for
for (int h = 0; h < numHalfEdges; ++h) {
if (twin(h) < 0) {
boundaryEdgePoint(mesh, h);
} else if (twin(h) > h) {
interiorEdgePoint(mesh, h);
interiorEdgePoint(mesh, twin(h));
}
}

#pragma omp for
for (int h = 0; h < numHalfEdges; ++h) {
float n = valence(h);
if (n > 0) {
interiorVertexPoint(mesh, h, n);
} else if (twin(h) < 0) {
boundaryVertexPoint(mesh, h);
}
}
}
}
