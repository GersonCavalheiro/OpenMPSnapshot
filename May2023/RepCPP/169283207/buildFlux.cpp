#include <iostream>
#include <cmath>
#include "buildFlux.hpp"
#include <Eigen/Dense>


void buildFlux(const Mesh& mesh, Field& field, double factor, double t,
const SolverParams& solverParams)
{
#pragma omp parallel for default(none) shared(field, mesh, solverParams, factor, t)
for(size_t elm = 0 ; elm < mesh.elements.size() ; elm++)
{
PartialField partialField(solverParams.nUnknowns, mesh.dim);

for(unsigned short unk = 0 ; unk < solverParams.nUnknowns ; ++unk)
{
partialField.partialIu[unk]
.resize(mesh.elementProperties
.at(mesh.elements[elm].elementTypeHD).nSF);
partialField.partialIu[unk].setZero();
}

unsigned int nSigma = mesh.elements[elm].edges.size();
for(unsigned int s = 0 ; s < nSigma ; ++s)
{

for(unsigned short dim = 0 ; dim < mesh.dim ; ++dim)
{
for(unsigned short unk = 0 ; unk < solverParams.nUnknowns ; ++unk)
{
partialField.g[dim][unk]
.resize(mesh.elementProperties
.at(mesh.elements[elm].elementTypeHD).nSF);
partialField.g[dim][unk].setZero();
}
}

for(unsigned int j = 0 ;
j < mesh.elements[elm].edges[s].offsetInElm.size() ; ++j)
{
unsigned int indexJ = mesh.elements[elm].offsetInU
+ mesh.elements[elm].edges[s].offsetInElm[j];

if (mesh.elements[elm].edges[s].edgeInFront.first == -1)
{
ibc boundary = solverParams.boundaryConditions
.at(mesh.elements[elm].edges[s].bcName);

boundary.ibcFunc(partialField.uAtBC,
mesh.elements[elm].edges[s].nodeCoordinate[j], t,
field, indexJ, mesh.elements[elm].edges[s].normal,
boundary.coefficients, solverParams.fluxCoeffs);

solverParams.flux(field, partialField, solverParams, true);

solverParams.phiPsi(mesh.elements[elm].edges[s], field,
partialField, j, factor, true, indexJ, 0, solverParams);
}
else 
{
unsigned int indexFrontJ =
mesh
.elements[mesh.elements[elm].edges[s]
.edgeInFront.first].offsetInU
+ mesh
.elements[mesh.elements[elm].edges[s]
.edgeInFront.first]
.edges[mesh.elements[elm].edges[s]
.edgeInFront.second]
.offsetInElm[mesh.elements[elm]
.edges[s]
.nodeIndexEdgeInFront[j]];

solverParams.phiPsi(mesh.elements[elm].edges[s], field,
partialField, j, factor, false, indexJ,
indexFrontJ, solverParams);
}
}

for(unsigned short unk = 0 ; unk < solverParams.nUnknowns ; ++unk)
{
partialField.partialIu[unk] +=
mesh.elements[elm].edges[s].determinantLD[0]*(
mesh.elements[elm].edges[s].normal[0]*mesh
.elements[elm].dM[s]*partialField.g[0][unk]
+ mesh.elements[elm].edges[s].normal[1]*mesh
.elements[elm].dM[s]*partialField.g[1][unk]);
}
}

for(unsigned short unk = 0 ; unk < field.Iu.size() ; ++unk)
{
for(unsigned int j = 0 ;
j < mesh.elementProperties
.at(mesh.elements[elm].elementTypeHD).nSF ; ++j)
{
field.Iu[unk][mesh.elements[elm].offsetInU + j]
= partialField.partialIu[unk][j];
}
}
}
}
