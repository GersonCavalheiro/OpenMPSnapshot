

#include <algorithm>
#include <iostream>
#include <fstream>
#include <gmsh.h>
#include "Mesh.hpp"
#include "../utils/utils.hpp"


static void loadNodeData(Mesh& mesh)
{

unsigned int numNodes = 0;
std::vector<std::size_t> elementTags;
std::vector<unsigned int> elementNumNodes;
std::vector<std::size_t> nodeTags;
std::vector<std::vector<double>> coord;

for(size_t elm = 0 ; elm < mesh.elements.size() ; ++elm)
{
elementTags.push_back(mesh.elements[elm].elementTag);
elementNumNodes.push_back(mesh.elements[elm].nodeTags.size());

for(size_t n = 0 ; n < mesh.elements[elm].nodeTags.size() ; ++n)
{
std::vector<double> temp;
temp.push_back(mesh.elements[elm].nodesCoord[n][0]);
temp.push_back(mesh.elements[elm].nodesCoord[n][1]);
coord.push_back(temp);

nodeTags.push_back(mesh.elements[elm].nodeTags[n]);
numNodes++;

}
}

mesh.nodeData.numNodes = numNodes;
mesh.nodeData.elementTags = elementTags;
mesh.nodeData.elementNumNodes = elementNumNodes;
mesh.nodeData.nodeTags = nodeTags;
mesh.nodeData.coord = coord;
}



static void loadElementProperties(std::map<int, ElementProperty>& meshElementProp,
const std::vector<int>& eleTypes,
const std::string& intScheme,
const std::string& basisFuncType)
{
for(unsigned int i = 0 ; i < eleTypes.size() ; ++i)
{
if(meshElementProp.count(eleTypes[i]) == 0)
{
ElementProperty elementProperty;
gmsh::model::mesh::getElementProperties(eleTypes[i],
elementProperty.name,
elementProperty.dim,
elementProperty.order,
elementProperty.numNodes,
elementProperty.paramCoord);

gmsh::model::mesh::getIntegrationPoints(eleTypes[i], intScheme,
elementProperty.intPoints,
elementProperty.intWeigths);

gmsh::model::mesh::getBasisFunctions(eleTypes[i],
elementProperty.intPoints,
basisFuncType,
elementProperty.numComp,
elementProperty.basisFunc);

int dummyNumComp;
gmsh::model::mesh::getBasisFunctions(eleTypes[i],
elementProperty.intPoints,
std::string("Grad" + basisFuncType),
dummyNumComp,
elementProperty.basisFuncGrad);


elementProperty.nGP = elementProperty.intPoints.size()/3;
elementProperty.nSF = elementProperty.basisFunc.size()
/elementProperty.nGP;
for(unsigned int k = 0 ; k < elementProperty.nGP ; ++k)
{
std::vector<double> wll;
std::vector<double> wl;

for(unsigned int i = 0 ; i < elementProperty.nSF ; ++i)
{

wl.push_back(elementProperty.intWeigths[k]
*elementProperty.basisFunc[elementProperty.nSF*k + i]);

for(unsigned int j = i ; j < elementProperty.nSF ; ++j)
{
if(k == 0)
{
elementProperty.IJ.push_back(
std::pair<unsigned int, unsigned int>(i, j));
}

wll.push_back(elementProperty.intWeigths[k]
*elementProperty.basisFunc[elementProperty.nSF*k + i]
*elementProperty.basisFunc[elementProperty.nSF*k + j]);
}
}

elementProperty.pondFunc.push_back(wl);
elementProperty.prodFunc.push_back(wll);
}

std::vector<std::vector<double>> lalb(elementProperty.nSF);
for(size_t l = 0 ; l < lalb.size() ; ++l)
{
lalb[l].resize(elementProperty.nSF);
}

for(unsigned int l = 0 ;
l < elementProperty.nSF*(elementProperty.nSF+1)/2 ; ++l)
{
double  sum = 0.0;
for (unsigned int k = 0 ; k < elementProperty.nGP ; ++k)
{
sum += elementProperty.prodFunc[k][l];
}

lalb[elementProperty.IJ[l].first][elementProperty.IJ[l].second] = sum;
if(elementProperty.IJ[l].first != elementProperty.IJ[l].second)
{
lalb[elementProperty.IJ[l].second][elementProperty.IJ[l].first]
= sum;
}
}

elementProperty.lalb = std::move(lalb);

meshElementProp[eleTypes[i]] = std::move(elementProperty);
}
}
}



static void addEdge(Element& element, std::vector<int> nodesTagsEdge,
std::vector<double> determinantLD,
const std::vector<int>& nodesTagsElement)
{

Edge edge;
for(unsigned int i = 0 ; i < nodesTagsEdge.size() ; ++i)
{
for(unsigned int j = 0 ; j < nodesTagsElement.size() ; ++j)
{
if(nodesTagsEdge[i] == nodesTagsElement[j]) 
edge.offsetInElm.push_back(j);

}
}
edge.nodeTags = std::move(nodesTagsEdge);
edge.determinantLD = std::move(determinantLD);

element.edges.push_back(edge);
}



static void computeEdgeNormalCoord(Edge& edge, unsigned int meshDim,
const std::vector<double>& baryCenter)
{
std::vector<double> normal;

std::vector<double> coord1, coord2;
for(unsigned int i = 0 ; i < edge.nodeTags.size() ; ++i)
{
std::vector<double> coord, dummyParametricCoord;
gmsh::model::mesh::getNode(edge.nodeTags[i], coord,
dummyParametricCoord);

edge.nodeCoordinate.push_back(coord);

if(i == 0)
coord1 = std::move(coord);
else if(i == 1)
coord2 = std::move(coord);
}

edge.length=sqrt((coord1[0]-coord2[0])*(coord1[0]-coord2[0])
+(coord1[1]-coord2[1])*(coord1[1]-coord2[1])
+(coord1[2]-coord2[2])*(coord1[2]-coord2[2]));

switch(meshDim)
{
case 1:
if(coord1[0]-baryCenter[0] < 0)
normal.push_back(-1);
else
normal.push_back(1);
edge.normal = std::move(normal);
break;

case 2:
double nx = coord2[1] - coord1[1];
double ny = coord1[0] - coord2[0];
double norm = sqrt(ny*ny + nx*nx);

double vx = baryCenter[0] - (coord2[0] + coord1[0])/2;
double vy = baryCenter[1] - (coord2[1] + coord1[1])/2;

if(nx*vx + ny*vy > 0)
{
nx = -nx;
ny = -ny;
}

normal.push_back(nx/norm);
normal.push_back(ny/norm);

edge.normal = std::move(normal);
break;
}
}


static void findInFrontEdge(Mesh& mesh, Edge& currentEdge, unsigned int edgePos)
{
unsigned int elVecSize = mesh.elements.size();
unsigned int nEdgePerEl = mesh.elements[0].edges.size();

#pragma omp parallel default(none) shared(mesh, elVecSize, nEdgePerEl, currentEdge, edgePos)
{
#pragma omp for
for(unsigned int elm = 0 ; elm < elVecSize ; ++elm)
{
for(unsigned int k = 0 ; k < nEdgePerEl ; ++k)
{
std::vector<int> nodesTagsInfront
= mesh.elements[elm].edges[k].nodeTags;
std::vector<unsigned int> permutation1, permutation2;

if(isPermutation(currentEdge.nodeTags, nodesTagsInfront,
permutation1, permutation2))
{
#pragma omp critical
{
currentEdge.edgeInFront
=  std::pair<unsigned int, unsigned int>(elm, k);
mesh.elements[elm].edges[k].edgeInFront
= std::pair<unsigned int, unsigned int>
(elVecSize, edgePos);
currentEdge.nodeIndexEdgeInFront = std::move(permutation1);
mesh.elements[elm].edges[k].nodeIndexEdgeInFront
= std::move(permutation2);
}
#pragma omp cancel for
}
}
#pragma omp cancellation point for
}
}
}


static bool IsBoundary(const std::map<std::string,
std::vector<std::size_t>>& nodesTagBoundaries, Edge& edge)
{
for(std::pair<std::string, std::vector<std::size_t>> nodeTagBoundary : nodesTagBoundaries)
{
for(unsigned int j = 0 ; j < edge.nodeTags.size() ; ++j)
{
if(!std::count(nodeTagBoundary.second.begin(),
nodeTagBoundary.second.end(), edge.nodeTags[j]))
break;

if(j == edge.nodeTags.size() - 1)
{
edge.bcName = nodeTagBoundary.first;
return true;
}
}
}
return false;
}



static void addElement(Mesh& mesh, int elementTag, int eleTypeHD,
int eleTypeLD, std::vector<double> jacobiansHD,
std::vector<double> determinantsHD,
std::vector<double> determinantsLD,
unsigned int nGPLD, unsigned int offsetInU,
std::vector<int> nodesTagsPerEdge,
std::vector<int> nodesTags,
const std::vector<double>& elementBarycenter)
{
Element element;
element.elementTag = elementTag;
element.elementTypeHD = eleTypeHD;
element.elementTypeLD = eleTypeLD;

element.offsetInU = offsetInU;

element.determinantHD = std::move(determinantsHD);
element.jacobianHD = std::move(jacobiansHD);
element.nodeTags = std::move(nodesTags);

for(unsigned int i = 0 ; i < element.nodeTags.size() ; ++i)
{
std::vector<double> coord, dummyParametricCoord;
gmsh::model::mesh::getNode(element.nodeTags[i], coord, dummyParametricCoord);

element.nodesCoord.push_back(coord);
}

unsigned int nEdge = determinantsLD.size()/nGPLD;
unsigned int nNodePerEdge = mesh.elementProperties.at(eleTypeLD).numNodes;

for(unsigned int i = 0 ; i < nEdge ; ++i)
{
std::vector<int> nodesTagsEdge(nodesTagsPerEdge.begin() + nNodePerEdge*i,
nodesTagsPerEdge.begin()
+ nNodePerEdge*(i + 1));

std::vector<double> determinantsEdgeLD(determinantsLD.begin()
+ nGPLD*i, determinantsLD.begin()
+ nGPLD*(i + 1));

addEdge(element, std::move(nodesTagsEdge), std::move(determinantsEdgeLD),
element.nodeTags);
if(mesh.elements.size() != 0)
{
if(!IsBoundary(mesh.nodesTagBoundary, element.edges[i]))
findInFrontEdge(mesh, element.edges[i], i);

}
else
{
IsBoundary(mesh.nodesTagBoundary, element.edges[i]);
}

computeEdgeNormalCoord(element.edges[i], mesh.dim, elementBarycenter);

Eigen::SparseMatrix<double> dMs(mesh.elementProperties.at(eleTypeHD).nSF,
mesh.elementProperties.at(eleTypeHD).nSF);
std::vector<Eigen::Triplet<double>> indices;

for(size_t nA = 0 ; nA <  element.edges[i].nodeTags.size() ; ++nA)
{
for(size_t nB = 0 ; nB <  element.edges[i].nodeTags.size() ; ++nB)
{
indices.push_back(Eigen::Triplet<double>
(element.edges[i].offsetInElm[nA],
element.edges[i].offsetInElm[nB],
mesh.elementProperties.at(eleTypeLD).lalb[nA][nB]));
}
}
dMs.setFromTriplets(indices.begin(), indices.end());
element.dM.push_back(dMs);
}

mesh.elements.push_back(element);
}



static bool buildMesh(Mesh& mesh, int entityTag, unsigned int& currentOffset,
const std::string& intScheme, const std::string& basisFuncType)
{
mesh.entityTagHD = entityTag;

int c = gmsh::model::addDiscreteEntity(1);
mesh.entityTagLD = c;

std::vector<int> eleTypesHD;
gmsh::model::mesh::getElementTypes(eleTypesHD, mesh.dim,entityTag);

loadElementProperties(mesh.elementProperties, eleTypesHD,
intScheme, basisFuncType);

for(auto eleTypeHD : eleTypesHD)
{
std::vector<std::size_t> nodeTags, elementTags;
gmsh::model::mesh::getElementsByType(eleTypeHD, elementTags, nodeTags,
entityTag);

std::vector<std::size_t> nodesTagPerEdge;
gmsh::model::mesh::getElementEdgeNodes(eleTypeHD, nodesTagPerEdge,
entityTag);

unsigned int numNodes = mesh.elementProperties[eleTypeHD].numNodes;
unsigned int order = mesh.elementProperties[eleTypeHD].order;

if(intScheme == "Lagrange" && order > 7)
{
std::cerr << "Lagrange polynomials are not stable with an order"
<< " superior to 7"
<< std::endl;

return false;
}

std::vector<double> baryCenters;
gmsh::model::mesh::getBarycenters(eleTypeHD, entityTag, false, true,
baryCenters);

int eleTypeLD;
switch(mesh.dim)
{
case 1:
eleTypeLD = gmsh::model::mesh::getElementType("point", order);
break;

case 2:
eleTypeLD = gmsh::model::mesh::getElementType("line", order);
break;
}

gmsh::model::mesh::addElementsByType(c, eleTypeLD, {}, nodesTagPerEdge);

loadElementProperties(mesh.elementProperties, std::vector<int>(1, eleTypeLD),
intScheme, basisFuncType);

std::vector<double> jacobiansHD, determinantsHD, dummyPointsHD;
gmsh::model::mesh::getJacobians(eleTypeHD,
mesh.elementProperties[eleTypeHD].intPoints,
jacobiansHD, determinantsHD, dummyPointsHD,
entityTag);

std::vector<double> dummyJacobiansLD, determinantsLD, dummyPointsLD;
gmsh::model::mesh::getJacobians(eleTypeLD,
mesh.elementProperties[eleTypeLD].intPoints,
dummyJacobiansLD, determinantsLD,
dummyPointsLD, c);

unsigned int nElements = nodeTags.size()
/mesh.elementProperties[eleTypeHD].numNodes;
unsigned int nGPHD = mesh.elementProperties[eleTypeHD].nGP;
unsigned int nGPLD = mesh.elementProperties[eleTypeLD].nGP;
unsigned int nEdgePerElement = determinantsLD.size()/(nGPLD*nElements);
unsigned int nNodesPerEdge = mesh.elementProperties[eleTypeLD].numNodes;

unsigned ratio, currentDecade = 0;
for(unsigned int i = 0 ; i < elementTags.size() ; ++i)
{

ratio = int(100*double(i)/double(elementTags.size()));
if(ratio >= currentDecade)
{
std::cout   << "\r" << "Entity [" << mesh.entityTagHD << "]: "
<< ratio << "% of the elements computed"
<< std::flush;
currentDecade = ratio + 1;
}

std::vector<double> jacobiansElementHD(
jacobiansHD.begin() + 9*nGPHD*i,
jacobiansHD.begin() + 9*nGPHD*(1 + i));
std::vector<double> determinantsElementHD(
determinantsHD.begin() + nGPHD*i,
determinantsHD.begin() + nGPHD*(1 + i));
std::vector<double> determinantElementLD(
determinantsLD.begin() + nEdgePerElement
*nGPLD*i,
determinantsLD.begin() + nEdgePerElement
*nGPLD*(1 + i));

std::vector<int> nodeTagsElement(nodeTags.begin() + numNodes*i,
nodeTags.begin() + numNodes*(1 + i));

std::vector<int> nodesTagPerEdgeElement(
nodesTagPerEdge.begin()
+ nNodesPerEdge*nEdgePerElement*i,
nodesTagPerEdge.begin()
+ nNodesPerEdge*nEdgePerElement*(i + 1));

std::vector<double> elementBarycenter(baryCenters.begin() + 3*i,
baryCenters.begin() + 3*(i + 1));

unsigned int elementOffset = numNodes;

double dx;

addElement(mesh, elementTags[i], eleTypeHD, eleTypeLD,
std::move(jacobiansElementHD),
std::move(determinantsElementHD),
std::move(determinantElementLD),
nGPLD, currentOffset,
std::move(nodesTagPerEdgeElement),
std::move(nodeTagsElement),
elementBarycenter);

currentOffset += elementOffset;
}

std::cout  << "\r" << "Entity [" << mesh.entityTagHD << "]: "
<< "100% of the elements computed" << std::flush << std::endl;
}

return true;
}



static unsigned short getMeshDim()
{
int elementDim = -1;

for(unsigned short i = 0 ; i <= 3 ; ++i)
{
std::vector<int> eleTypes;
gmsh::model::mesh::getElementTypes(eleTypes, i);

switch(eleTypes.size())
{
case 0:
break;
case 1:
elementDim = i;
break;
default:
elementDim = i;
std::cerr   << "Hybrid meshes not handled in this example!"
<< std::endl;
}
}

return elementDim;
}

std::vector<int> getTags(const Mesh& mesh)
{
std::vector<int> listTags;

for(unsigned int elm = 0 ; elm < mesh.elements.size() ; ++elm)
{
for(unsigned int n = 0 ; n < mesh.elements[elm].nodeTags.size() ; ++n)
{
listTags.push_back(mesh.elements[elm].nodeTags[n]);
}
}

return listTags;
}

bool readMesh(Mesh& mesh, const std::string& fileName,
const std::string& intScheme, const std::string& basisFuncType)
{
gmsh::initialize();
gmsh::option::setNumber("General.Terminal", 1);
std::ifstream file(fileName);
if(file.is_open())
file.close();
else
{
std::cerr << "File: " << fileName << " does not exist!" << std::endl;
return false;
}
gmsh::open(fileName);

mesh.dim = getMeshDim();
if(mesh.dim == 3)
{
std::cerr << "3D meshes unsupported!" << std::endl;
return false;
}

std::vector<std::pair<int, int>> physGroupHandles;
gmsh::model::getPhysicalGroups(physGroupHandles, mesh.dim);

std::vector<std::pair<int, int>> BCHandles;
gmsh::model::getPhysicalGroups(BCHandles, mesh.dim - 1);

for(auto BCHandle : BCHandles)
{
std::string name;
gmsh::model::getPhysicalName(mesh.dim - 1, BCHandle.second, name);
std::vector<std::size_t> nodesTags;
std::vector<double> dummyCoord;
gmsh::model::mesh::getNodesForPhysicalGroup(mesh.dim - 1, BCHandle.second,
nodesTags, dummyCoord);
mesh.nodesTagBoundary[name] = nodesTags;
}

std::vector<int> entitiesTag;
for(auto physGroupHandle : physGroupHandles)
{
std::vector<int> entityTag;
gmsh::model::getEntitiesForPhysicalGroup(physGroupHandle.first,
physGroupHandle.second,
entityTag);

entitiesTag.push_back(entityTag[0]);
}


unsigned int currentOffset = 0;

if(entitiesTag.size() > 1)
{
std::cerr << "Multiple " << mesh.dim << "D entities in " << mesh.dim
<< "D meshes " << "currently not supported" <<std::endl;

return false;
}

for(auto entityTag : entitiesTag)
{
if(!buildMesh(mesh, entityTag, currentOffset, intScheme, basisFuncType))
return false;
}

loadNodeData(mesh);

gmsh::finalize();

return true;
}
