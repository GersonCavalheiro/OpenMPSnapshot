
#include "includes/model_part.h"

#if !defined(KRATOS_BOUNDARY_NORMALS_CALCULATION_UTILITIES_H_INCLUDED)
#define KRATOS_BOUNDARY_NORMALS_CALCULATION_UTILITIES_H_INCLUDED



#include "delaunay_meshing_application_variables.h"
#include "custom_utilities/mesher_utilities.hpp"

namespace Kratos
{






class BoundaryNormalsCalculationUtilities
{
public:

typedef ModelPart::NodesContainerType NodesArrayType;
typedef ModelPart::ElementsContainerType ElementsContainerType;
typedef ModelPart::ConditionsContainerType ConditionsContainerType;
typedef ModelPart::MeshType MeshType;

typedef GlobalPointersVector<Node> NodeWeakPtrVectorType;
typedef GlobalPointersVector<Element> ElementWeakPtrVectorType;
typedef GlobalPointersVector<Condition> ConditionWeakPtrVectorType;





void CalculateUnitBoundaryNormals(ModelPart &rModelPart, int EchoLevel = 0)
{

mEchoLevel = EchoLevel;

if (!rModelPart.IsSubModelPart())
{

this->ResetBodyNormals(rModelPart); 

for (auto &i_mp : rModelPart.SubModelParts())
{
if ((i_mp.Is(FLUID) || i_mp.Is(SOLID) || i_mp.Is(RIGID)) && i_mp.IsNot(ACTIVE) && i_mp.IsNot(BOUNDARY) && i_mp.IsNot(CONTACT))
{

CalculateBoundaryNormals(i_mp);

AddNormalsToNodes(i_mp);
}
}

}
else
{

this->ResetBodyNormals(rModelPart); 

CalculateBoundaryNormals(rModelPart);

AddNormalsToNodes(rModelPart);
}

rModelPart.GetCommunicator().AssembleCurrentData(NORMAL);
}

void CalculateWeightedBoundaryNormals(ModelPart &rModelPart, int EchoLevel = 0)
{

std::cout << "DO NOT ENTER HERE ----   CalculateWeightedBoundaryNormals " << std::endl;
mEchoLevel = EchoLevel;

if (!rModelPart.IsSubModelPart())
{

this->ResetBodyNormals(rModelPart); 

for (auto &i_mp : rModelPart.SubModelParts())
{
if (i_mp.Is(FLUID) && i_mp.IsNot(ACTIVE) && i_mp.IsNot(BOUNDARY) && i_mp.IsNot(CONTACT))
{

CalculateBoundaryNormals(i_mp);

AddWeightedNormalsToNodes(i_mp);
}
}
for (auto &i_mp : rModelPart.SubModelParts())
{
if (i_mp.Is(SOLID) && i_mp.IsNot(ACTIVE) && i_mp.IsNot(BOUNDARY) && i_mp.IsNot(CONTACT))
{

CalculateBoundaryNormals(i_mp);

AddWeightedNormalsToNodes(i_mp);
}
}
for (auto &i_mp : rModelPart.SubModelParts())
{
if (i_mp.Is(RIGID) && i_mp.IsNot(ACTIVE) && i_mp.IsNot(BOUNDARY) && i_mp.IsNot(CONTACT))
{

CalculateBoundaryNormals(i_mp);

AddWeightedNormalsToNodes(i_mp);
}
}
}
else
{

this->ResetBodyNormals(rModelPart); 

CalculateBoundaryNormals(rModelPart);

}

rModelPart.GetCommunicator().AssembleCurrentData(NORMAL);
rModelPart.GetCommunicator().AssembleCurrentData(SHRINK_FACTOR);
}

protected:

private:

int mEchoLevel;


void CalculateBoundaryNormals(ModelPart &rModelPart)
{
KRATOS_TRY

const unsigned int dimension = rModelPart.GetProcessInfo()[SPACE_DIMENSION];

if (rModelPart.NumberOfConditions() && this->CheckConditionsLocalSpace(rModelPart, dimension - 1))
{

if (mEchoLevel > 0)
std::cout << "   [" << rModelPart.Name() << "] (BC)" << std::endl;

this->CalculateBoundaryNormals(rModelPart.Conditions());
}
else if (rModelPart.NumberOfElements())
{

if (this->CheckElementsLocalSpace(rModelPart, dimension))
{

if (mEchoLevel > 0)
std::cout << "   [" << rModelPart.Name() << "] (BVE)" << std::endl;

this->CalculateVolumeBoundaryNormals(rModelPart);
}
else if (this->CheckElementsLocalSpace(rModelPart, dimension - 1))
{

if (mEchoLevel > 0)
std::cout << "   [" << rModelPart.Name() << "] (BE)" << std::endl;

ElementsContainerType &rElements = rModelPart.Elements();

this->CalculateBoundaryNormals(rElements);
}
}

KRATOS_CATCH("")
}

static void CalculateUnitNormal2D(Condition &rCondition, array_1d<double, 3> &An)
{
Geometry<Node> &rGeometry = rCondition.GetGeometry();

An[0] = rGeometry[1].Y() - rGeometry[0].Y();
An[1] = -(rGeometry[1].X() - rGeometry[0].X());
An[2] = 0.00;

array_1d<double, 3> &normal = rCondition.GetValue(NORMAL);
noalias(normal) = An / norm_2(An);
}

static void CalculateUnitNormal3D(Condition &rCondition, array_1d<double, 3> &An,
array_1d<double, 3> &v1, array_1d<double, 3> &v2)
{
Geometry<Node> &rGeometry = rCondition.GetGeometry();

v1[0] = rGeometry[1].X() - rGeometry[0].X();
v1[1] = rGeometry[1].Y() - rGeometry[0].Y();
v1[2] = rGeometry[1].Z() - rGeometry[0].Z();

v2[0] = rGeometry[2].X() - rGeometry[0].X();
v2[1] = rGeometry[2].Y() - rGeometry[0].Y();
v2[2] = rGeometry[2].Z() - rGeometry[0].Z();

MathUtils<double>::CrossProduct(An, v1, v2);

array_1d<double, 3> &normal = rCondition.GetValue(NORMAL);

noalias(normal) = An / norm_2(An);
}

static void CalculateUnitNormal2D(Element &rElement, array_1d<double, 3> &An)
{
Geometry<Node> &rGeometry = rElement.GetGeometry();

if (rGeometry.size() < 2)
{
std::cout << " Warning 2D geometry with only " << rGeometry.size() << " node :: multiple normal definitions " << std::endl;
rElement.GetValue(NORMAL).clear();
}
else
{

An[0] = -(rGeometry[1].Y() - rGeometry[0].Y());
An[1] = rGeometry[1].X() - rGeometry[0].X();
An[2] = 0.00;

array_1d<double, 3> &normal = rElement.GetValue(NORMAL);
noalias(normal) = An / norm_2(An);
}
}

static void CalculateUnitNormal3D(Element &rElement, array_1d<double, 3> &An,
array_1d<double, 3> &v1, array_1d<double, 3> &v2)
{
Geometry<Node> &rGeometry = rElement.GetGeometry();

if (rGeometry.size() < 3)
{
std::cout << " Warning 3D geometry with only " << rGeometry.size() << " nodes :: multiple normal definitions " << std::endl;
rElement.GetValue(NORMAL).clear();
}
else
{

v1[0] = rGeometry[1].X() - rGeometry[0].X();
v1[1] = rGeometry[1].Y() - rGeometry[0].Y();
v1[2] = rGeometry[1].Z() - rGeometry[0].Z();

v2[0] = rGeometry[2].X() - rGeometry[0].X();
v2[1] = rGeometry[2].Y() - rGeometry[0].Y();
v2[2] = rGeometry[2].Z() - rGeometry[0].Z();

MathUtils<double>::CrossProduct(An, v1, v2);
An *= 0.5;

array_1d<double, 3> &normal = rElement.GetValue(NORMAL);

noalias(normal) = An / norm_2(An);
}
}

void ResetBodyNormals(ModelPart &rModelPart)
{
for (auto &i_node : rModelPart.Nodes())
{
i_node.GetSolutionStepValue(NORMAL).clear();
}
}

void CheckBodyNormals(ModelPart &rModelPart)
{
for (const auto &i_node : rModelPart.Nodes())
{
std::cout << " ID: " << i_node.Id() << " normal: " << i_node.GetSolutionStepValue(NORMAL) << std::endl;
}
}


void CalculateBoundaryNormals(ConditionsContainerType &rConditions)
{
KRATOS_TRY


const unsigned int dimension = rConditions.begin()->GetGeometry().WorkingSpaceDimension();


array_1d<double, 3> An;
if (dimension == 2)
{
for (auto &i_cond : rConditions)
{
if (i_cond.IsNot(CONTACT) && i_cond.Is(BOUNDARY))
CalculateUnitNormal2D(i_cond, An);
}
}
else if (dimension == 3)
{
array_1d<double, 3> v1;
array_1d<double, 3> v2;
for (auto &i_cond : rConditions)
{
if (i_cond.IsNot(CONTACT) && i_cond.Is(BOUNDARY))
{
CalculateUnitNormal3D(i_cond, An, v1, v2);
}
}
}

KRATOS_CATCH("")
}


void CalculateBoundaryNormals(ElementsContainerType &rElements)

{
KRATOS_TRY


const unsigned int dimension = (rElements.begin())->GetGeometry().WorkingSpaceDimension();


array_1d<double, 3> An;
if (dimension == 2)
{
for (auto &i_elem : rElements)
{
if (i_elem.IsNot(CONTACT))
{
i_elem.Set(BOUNDARY); 
CalculateUnitNormal2D(i_elem, An);
}
}
}
else if (dimension == 3)
{
array_1d<double, 3> v1;
array_1d<double, 3> v2;
for (auto &i_elem : rElements)
{
if (i_elem.IsNot(CONTACT))
{
i_elem.Set(BOUNDARY); 
CalculateUnitNormal3D(i_elem, An, v1, v2);
}
}
}

KRATOS_CATCH("")
}

bool CheckElementsDimension(ModelPart &rModelPart, unsigned int dimension)
{
KRATOS_TRY

ElementsContainerType &rElements = rModelPart.Elements();

ElementsContainerType::iterator it = rElements.begin();

if ((it)->GetGeometry().WorkingSpaceDimension() == dimension)
{
return true;
}
else
{
return false;
}

KRATOS_CATCH("")
}

bool CheckConditionsDimension(ModelPart &rModelPart, unsigned int dimension)
{
KRATOS_TRY

if (rModelPart.Conditions().begin()->GetGeometry().WorkingSpaceDimension() == dimension)
{
return true;
}
else
{
return false;
}

KRATOS_CATCH("")
}

bool CheckElementsLocalSpace(ModelPart &rModelPart, unsigned int dimension)
{
KRATOS_TRY

if (rModelPart.Elements().begin()->GetGeometry().LocalSpaceDimension() == dimension)
{
return true;
}
else
{
return false;
}

KRATOS_CATCH("")
}

bool CheckConditionsLocalSpace(ModelPart &rModelPart, unsigned int dimension)
{
KRATOS_TRY

if (rModelPart.Conditions().begin()->GetGeometry().LocalSpaceDimension() == dimension)
{
return true;
}
else
{
return false;
}

KRATOS_CATCH("")
}

void CalculateVolumeBoundaryNormals(ModelPart &rModelPart)
{
KRATOS_TRY

const unsigned int dimension = rModelPart.GetProcessInfo()[SPACE_DIMENSION];

ModelPart::NodesContainerType &rNodes = rModelPart.Nodes();
ModelPart::ElementsContainerType &rElements = rModelPart.Elements();

bool neighsearch = false;
unsigned int number_of_nodes = rElements.begin()->GetGeometry().PointsNumber();
for (unsigned int i = 0; i < number_of_nodes; ++i)
if ((rElements.begin()->GetGeometry()[i].GetValue(NEIGHBOUR_ELEMENTS)).size() > 1)
neighsearch = true;

if (!neighsearch)
std::cout << " WARNING :: Neighbour Search Not PERFORMED " << std::endl;

for (auto &i_node : rNodes)
{
i_node.GetSolutionStepValue(NORMAL).clear();
if (!neighsearch)
{
i_node.GetValue(NEIGHBOUR_ELEMENTS).clear();
i_node.Reset(BOUNDARY);
}
}

if (!neighsearch)
{
for (auto i_elem(rElements.begin()); i_elem != rElements.end(); ++i_elem)
{
Element::GeometryType &rGeometry = i_elem->GetGeometry();
for (unsigned int i = 0; i < rGeometry.size(); ++i)
{
rGeometry[i].GetValue(NEIGHBOUR_ELEMENTS).push_back(*i_elem.base());
}
}
}


Vector An(3);
Element::IntegrationMethod mIntegrationMethod = Element::GeometryDataType::IntegrationMethod::GI_GAUSS_1; 
int PointNumber = 0;                                                                                      
Matrix J;
Matrix InvJ;
double detJ;
Matrix DN_DX;

unsigned int assigned = 0;
unsigned int not_assigned = 0;
unsigned int boundary_nodes = 0;
for (auto &i_node : rNodes)
{
noalias(An) = ZeroVector(3);


ElementWeakPtrVectorType &nElements = i_node.GetValue(NEIGHBOUR_ELEMENTS);

for (auto &i_nelem : nElements)
{

Element::GeometryType &rGeometry = i_nelem.GetGeometry();

if (rGeometry.EdgesNumber() > 1 && rGeometry.LocalSpaceDimension() == dimension)
{

const Element::GeometryType::IntegrationPointsArrayType &integration_points = rGeometry.IntegrationPoints(mIntegrationMethod);
const Element::GeometryType::ShapeFunctionsGradientsType &DN_De = rGeometry.ShapeFunctionsLocalGradients(mIntegrationMethod);

J.resize(dimension, dimension, false);
J = rGeometry.Jacobian(J, PointNumber, mIntegrationMethod);

InvJ.clear();
detJ = 0;
if (dimension == 2)
{
MathUtils<double>::InvertMatrix2(J, InvJ, detJ);
}
else if (dimension == 3)
{
MathUtils<double>::InvertMatrix3(J, InvJ, detJ);
}
else
{
MathUtils<double>::InvertMatrix(J, InvJ, detJ);
}

DN_DX = prod(DN_De[PointNumber], InvJ);

double IntegrationWeight = integration_points[PointNumber].Weight() * detJ;

for (unsigned int i = 0; i < rGeometry.size(); ++i)
{
if (i_node.Id() == rGeometry[i].Id())
{

for (unsigned int d = 0; d < dimension; ++d)
{
An[d] += DN_DX(i, d) * IntegrationWeight;
}
}
}

}

if (norm_2(An) > 1e-12)
{
noalias(i_node.FastGetSolutionStepValue(NORMAL)) = An / norm_2(An);
assigned += 1;
if (!neighsearch)
{
i_node.Set(BOUNDARY);
}
}
else
{
(i_node.FastGetSolutionStepValue(NORMAL)).clear();
not_assigned += 1;
if (!neighsearch)
{
i_node.Set(BOUNDARY, false);
}
}
}

if (i_node.Is(BOUNDARY))
boundary_nodes += 1;

}

if (mEchoLevel > 0)
std::cout << "  [ Boundary_Normals  (Mesh Nodes:" << rNodes.size() << ")[Boundary nodes: " << boundary_nodes << " (SET:" << assigned << " / NOT_SET:" << not_assigned << ")] ]" << std::endl;

KRATOS_CATCH("")
}


void AddNormalsToNodes(ModelPart &rModelPart)
{
KRATOS_TRY

const unsigned int dimension = rModelPart.GetProcessInfo()[SPACE_DIMENSION];

if (rModelPart.NumberOfConditions() && this->CheckConditionsDimension(rModelPart, dimension - 1))
{


for (auto &i_cond : rModelPart.Conditions())
{
Geometry<Node> &rGeometry = i_cond.GetGeometry();
double coeff = 1.00 / rGeometry.size();
const array_1d<double, 3> &An = i_cond.GetValue(NORMAL);

for (unsigned int i = 0; i < rGeometry.size(); ++i)
{
noalias(rGeometry[i].FastGetSolutionStepValue(NORMAL)) += coeff * An;
}
}
}
else if (rModelPart.NumberOfElements() && this->CheckElementsDimension(rModelPart, dimension - 1))
{

for (auto &i_elem : rModelPart.Elements())
{
Geometry<Node> &rGeometry = i_elem.GetGeometry();
double coeff = 1.00 / rGeometry.size();
const array_1d<double, 3> &An = i_elem.GetValue(NORMAL);

for (unsigned int i = 0; i < rGeometry.size(); ++i)
{
noalias(rGeometry[i].FastGetSolutionStepValue(NORMAL)) += coeff * An;
}
}
}

KRATOS_CATCH("")
}


void AddWeightedNormalsToNodes(ModelPart &rModelPart)
{
KRATOS_TRY

const unsigned int dimension = rModelPart.GetProcessInfo()[SPACE_DIMENSION];

ModelPart::NodesContainerType &rNodes = rModelPart.Nodes();

unsigned int MaxNodeId = MesherUtilities::GetMaxNodeId(rModelPart);
std::vector<int> NodeNeighboursIds(MaxNodeId + 1);
std::fill(NodeNeighboursIds.begin(), NodeNeighboursIds.end(), 0);

if (mEchoLevel > 1)
std::cout << "   [" << rModelPart.Name() << "] [conditions:" << rModelPart.NumberOfConditions() << ", elements:" << rModelPart.NumberOfElements() << "] dimension: " << dimension << std::endl;

if (rModelPart.NumberOfConditions() && this->CheckConditionsLocalSpace(rModelPart, dimension - 1))
{

if (mEchoLevel > 0)
std::cout << "   [" << rModelPart.Name() << "] (C)" << std::endl;

std::vector<ConditionWeakPtrVectorType> Neighbours(rNodes.size() + 1);

unsigned int id = 1;
for (auto i_cond(rModelPart.Conditions().begin()); i_cond != rModelPart.Conditions().end(); ++i_cond)
{
if (i_cond->IsNot(CONTACT) && i_cond->Is(BOUNDARY))
{

Condition::GeometryType &rGeometry = i_cond->GetGeometry();

if (mEchoLevel > 2)
std::cout << " Condition ID " << i_cond->Id() << " id " << id << std::endl;

for (unsigned int i = 0; i < rGeometry.size(); ++i)
{
if (mEchoLevel > 2)
{
if (NodeNeighboursIds.size() <= rGeometry[i].Id())
std::cout << " Shrink node in geom " << rGeometry[i].Id() << " number of nodes " << NodeNeighboursIds.size() << std::endl;
}

if (NodeNeighboursIds[rGeometry[i].Id()] == 0)
{
NodeNeighboursIds[rGeometry[i].Id()] = id;
Neighbours[id].push_back(*i_cond.base());
id++;
}
else
{
Neighbours[NodeNeighboursIds[rGeometry[i].Id()]].push_back(*i_cond.base());
}
}
}
}

if (id > 1)
{
ModelPart::NodesContainerType::iterator nodes_begin = rNodes.begin();
ModelPart::NodesContainerType BoundaryNodes;

if (rModelPart.Is(FLUID))
{
for (unsigned int i = 0; i < rNodes.size(); ++i)
{
if ((nodes_begin + i)->Is(BOUNDARY) && (nodes_begin + i)->IsNot(RIGID) && NodeNeighboursIds[(nodes_begin + i)->Id()] != 0)
{
BoundaryNodes.push_back(*(nodes_begin + i).base());
}
}
}
else
{

for (unsigned int i = 0; i < rNodes.size(); ++i)
{
if ((nodes_begin + i)->Is(BOUNDARY) && NodeNeighboursIds[(nodes_begin + i)->Id()] != 0)
{
BoundaryNodes.push_back(*(nodes_begin + i).base());
}
}
}

ComputeBoundaryShrinkage<Condition>(BoundaryNodes, Neighbours, NodeNeighboursIds, dimension);
}
}
else if (rModelPart.NumberOfElements() && this->CheckElementsLocalSpace(rModelPart, dimension - 1))
{

if (mEchoLevel > 0)
std::cout << "   [" << rModelPart.Name() << "] (E) " << std::endl;

ModelPart::ElementsContainerType &rElements = rModelPart.Elements();

std::vector<ElementWeakPtrVectorType> Neighbours(rNodes.size() + 1);

unsigned int id = 1;
for (auto i_elem(rElements.begin()); i_elem != rElements.end(); ++i_elem)
{
if (i_elem->IsNot(CONTACT) && i_elem->Is(BOUNDARY))
{

Condition::GeometryType &rGeometry = i_elem->GetGeometry();

if (mEchoLevel > 2)
std::cout << " Element ID " << i_elem->Id() << " id " << id << std::endl;

for (unsigned int i = 0; i < rGeometry.size(); ++i)
{
if (mEchoLevel > 2)
{
if (NodeNeighboursIds.size() <= rGeometry[i].Id())
std::cout << " Shrink node in geom " << rGeometry[i].Id() << " number of nodes " << NodeNeighboursIds.size() << " Ids[id] " << NodeNeighboursIds[rGeometry[i].Id()] << std::endl;
}

if (NodeNeighboursIds[rGeometry[i].Id()] == 0)
{
NodeNeighboursIds[rGeometry[i].Id()] = id;
Neighbours[id].push_back(*i_elem.base());
id++;
}
else
{
Neighbours[NodeNeighboursIds[rGeometry[i].Id()]].push_back(*i_elem.base());
}
}
}
}

if (id > 1)
{
ModelPart::NodesContainerType::iterator nodes_begin = rNodes.begin();
ModelPart::NodesContainerType BoundaryNodes;

for (unsigned int i = 0; i < rNodes.size(); ++i)
{
if ((nodes_begin + i)->Is(BOUNDARY) && NodeNeighboursIds[(nodes_begin + i)->Id()] != 0)
{
BoundaryNodes.push_back(*((nodes_begin + i).base()));
}
}

ComputeBoundaryShrinkage<Element>(BoundaryNodes, Neighbours, NodeNeighboursIds, dimension);
}
}

KRATOS_CATCH("")
}


template <class TClassType>
void ComputeBoundaryShrinkage(ModelPart::NodesContainerType &rNodes, const std::vector<GlobalPointersVector<TClassType>> &rNeighbours, const std::vector<int> &rNodeNeighboursIds, const unsigned int &dimension)
{
KRATOS_TRY

ModelPart::NodesContainerType::iterator NodesBegin = rNodes.begin();
int NumberOfNodes = rNodes.size();

int not_assigned = 0;

#pragma omp parallel for reduction(+ \
: not_assigned)
for (int pn = 0; pn < NumberOfNodes; ++pn)
{
ModelPart::NodesContainerType::iterator iNode = NodesBegin + pn;
unsigned int Id = rNodeNeighboursIds[(iNode)->Id()];

double MeanCosinus = 0;

double TotalFaces = 0;
double ProjectionValue = 0;
double NumberOfNormals = 0;

int SingleFaces = 0;
int TipAcuteAngles = 0;

array_1d<double, 3> &rNormal = (iNode)->FastGetSolutionStepValue(NORMAL);
double &rShrinkFactor = (iNode)->FastGetSolutionStepValue(SHRINK_FACTOR);

unsigned int NumberOfNeighbourNormals = rNeighbours[Id].size();
if (NumberOfNeighbourNormals != 0)
{
noalias(rNormal) = ZeroVector(3);
rShrinkFactor = 0;
}

if (mEchoLevel > 1)
{

std::cout << " Id " << Id << " normals size " << NumberOfNeighbourNormals << " normal " << rNormal << " shrink " << rShrinkFactor << std::endl;
for (unsigned int i_norm = 0; i_norm < NumberOfNeighbourNormals; ++i_norm) 
{
std::cout << " normal [" << i_norm << "][" << (iNode)->Id() << "]: " << rNeighbours[Id][i_norm].GetValue(NORMAL) << std::endl;
}
}

Vector FaceNormals(NumberOfNeighbourNormals);
noalias(FaceNormals) = ZeroVector(NumberOfNeighbourNormals);
Vector TipNormals(NumberOfNeighbourNormals);
noalias(TipNormals) = ZeroVector(NumberOfNeighbourNormals);

for (unsigned int i_norm = 0; i_norm < NumberOfNeighbourNormals; ++i_norm) 
{

const array_1d<double, 3> &rEntityNormal = rNeighbours[Id][i_norm].GetValue(NORMAL); 

if (FaceNormals[i_norm] != 1)
{ 

double CloseProjection = 0;
for (unsigned int j_norm = i_norm + 1; j_norm < NumberOfNeighbourNormals; ++j_norm) 
{

const array_1d<double, 3> &rNormalVector = rNeighbours[Id][j_norm].GetValue(NORMAL); 

ProjectionValue = inner_prod(rEntityNormal, rNormalVector);

if (FaceNormals[j_norm] == 0)
{ 
if (ProjectionValue > 0.995)
{
FaceNormals[j_norm] = 1;
}
else if (ProjectionValue > 0.955)
{
CloseProjection += 1;
FaceNormals[j_norm] = 2;
FaceNormals[i_norm] = 2;
}
}

if (ProjectionValue < -0.005)
{
TipAcuteAngles += 1;
TipNormals[j_norm] += 1;
TipNormals[i_norm] += 1;
}

} 

for (unsigned int j_norm = i_norm; j_norm < NumberOfNeighbourNormals; ++j_norm) 
{
if (FaceNormals[j_norm] == 2 && CloseProjection > 0)
FaceNormals[j_norm] = (1.0 / (CloseProjection + 1));
}

if (FaceNormals[i_norm] != 0)
{                                                 
rNormal += rEntityNormal * FaceNormals[i_norm]; 
NumberOfNormals += FaceNormals[i_norm];
}
else
{
rNormal += rEntityNormal; 
NumberOfNormals += 1;
}

TotalFaces += 1;
}
}


if (dimension == 3 && NumberOfNormals >= 3 && TipAcuteAngles > 0)
{

std::vector<array_1d<double, 3>> NormalsTriad(3);
SingleFaces = 0;

if (NumberOfNormals == 3)
{ 

for (unsigned int i_norm = 0; i_norm < NumberOfNeighbourNormals; ++i_norm) 
{
if (TipNormals[i_norm] >= 1 && FaceNormals[i_norm] == 0)
{ 

const array_1d<double, 3> &rEntityNormal = rNeighbours[Id][i_norm].GetValue(NORMAL); 
NormalsTriad[SingleFaces] = rEntityNormal;
SingleFaces += 1;
}
}
}
else if (NumberOfNormals > 3)
{ 

std::vector<int> SignificativeNormals(NumberOfNeighbourNormals);
std::fill(SignificativeNormals.begin(), SignificativeNormals.end(), -1);

double MaxProjection = 0;
double Projection = 0;

for (unsigned int i_norm = 0; i_norm < NumberOfNeighbourNormals; ++i_norm) 
{
MaxProjection = std::numeric_limits<double>::min();

if (TipNormals[i_norm] >= 1 && FaceNormals[i_norm] != 1)
{ 

const array_1d<double, 3> &rEntityNormal = rNeighbours[Id][i_norm].GetValue(NORMAL); 

for (unsigned int j_norm = 0; j_norm < NumberOfNeighbourNormals; ++j_norm) 
{
if (TipNormals[j_norm] >= 1 && FaceNormals[j_norm] != 1 && i_norm != j_norm)
{ 

const array_1d<double, 3> &rNormalVector = rNeighbours[Id][j_norm].GetValue(NORMAL); 
Projection = inner_prod(rEntityNormal, rNormalVector);

if (MaxProjection < Projection && Projection > 0.3)
{ 
MaxProjection = Projection;
SignificativeNormals[i_norm] = j_norm; 
}
}
}
}
}


for (unsigned int i_norm = 0; i_norm < NumberOfNeighbourNormals; ++i_norm) 
{

if (SingleFaces < 3)
{

if (TipNormals[i_norm] >= 1 && FaceNormals[i_norm] != 1)
{ 

array_1d<double, 3> EntityNormal = rNeighbours[Id][i_norm].GetValue(NORMAL); 

if (FaceNormals[i_norm] != 0) 
EntityNormal *= FaceNormals[i_norm];

for (unsigned int j_norm = i_norm + 1; j_norm < NumberOfNeighbourNormals; ++j_norm) 
{
if (TipNormals[j_norm] >= 1 && FaceNormals[j_norm] != 1)
{ 

if (i_norm == (unsigned int)SignificativeNormals[j_norm])
{ 

const array_1d<double, 3> &rNormalVector = rNeighbours[Id][i_norm].GetValue(NORMAL); 

if (FaceNormals[i_norm] != 0)
{                                                      
EntityNormal += rNormalVector * FaceNormals[i_norm]; 
}
else
{
EntityNormal += rNormalVector;
}

if (norm_2(EntityNormal) != 0)
EntityNormal = EntityNormal / norm_2(EntityNormal);

FaceNormals[j_norm] = 1; 
}
}
}

NormalsTriad[SingleFaces] = EntityNormal;
++SingleFaces;
}
}
}
}

if (SingleFaces < 3)
{

for (unsigned int i_norm = 0; i_norm < NumberOfNeighbourNormals; ++i_norm) 
{
if (SingleFaces == 3)
break;

if (TipNormals[i_norm] >= 1 && FaceNormals[i_norm] != 0 && FaceNormals[i_norm] != 1)
{                                                                                      
const array_1d<double, 3> &rEntityNormal = rNeighbours[Id][i_norm].GetValue(NORMAL); 
NormalsTriad[SingleFaces] = rEntityNormal;
++SingleFaces;
}

if (TipNormals[i_norm] == 0 && FaceNormals[i_norm] == 0 && SingleFaces > 0)
{                                                                                      
const array_1d<double, 3> &rEntityNormal = rNeighbours[Id][i_norm].GetValue(NORMAL); 
NormalsTriad[SingleFaces] = rEntityNormal;
++SingleFaces;
}
}
}

if (SingleFaces == 2)
{

for (unsigned int i_norm = 0; i_norm < NumberOfNeighbourNormals; ++i_norm) 
{
if (SingleFaces == 3)
break;

if (TipNormals[i_norm] == 0 && FaceNormals[i_norm] != 0 && FaceNormals[i_norm] != 1 && SingleFaces > 0)
{ 
NormalsTriad[SingleFaces] = rNormal;
++SingleFaces;
}
}
}

if (SingleFaces == 3)
{

array_1d<double, 3> CrossProductN;
MathUtils<double>::CrossProduct(CrossProductN, NormalsTriad[1], NormalsTriad[2]);
double Projection = inner_prod(NormalsTriad[0], CrossProductN);

if (fabs(Projection) > 1e-15)
{
rNormal = CrossProductN;
MathUtils<double>::CrossProduct(CrossProductN, NormalsTriad[2], NormalsTriad[0]);
rNormal += CrossProductN;
MathUtils<double>::CrossProduct(CrossProductN, NormalsTriad[0], NormalsTriad[1]);
rNormal += CrossProductN;

rNormal /= Projection; 
}
else
{
rNormal = 0.5 * (NormalsTriad[1] + NormalsTriad[2]);
}
}
else if (SingleFaces == 2)
{
rNormal = 0.5 * (NormalsTriad[0] + NormalsTriad[1]);
}
else
{
std::cout << " WARNING something was wrong in normal calculation Triad Not found" << std::endl;
std::cout << " Face: " << FaceNormals << " Tip: " << TipNormals << " Id: " << (iNode)->Id() << " NumberOfNormals " << NumberOfNormals << " SingleFaces " << SingleFaces << std::endl;
}

if (norm_2(rNormal) > 2)
{ 
rNormal = rNormal / norm_2(rNormal);
rNormal *= 1.2;
}




}
else
{

if (norm_2(rNormal) != 0)
rNormal = rNormal / norm_2(rNormal);

for (unsigned int i_norm = 0; i_norm < NumberOfNeighbourNormals; ++i_norm) 
{
if (FaceNormals[i_norm] != 1)
{ 

array_1d<double, 3> rEntityNormal = rNeighbours[Id][i_norm].GetValue(NORMAL); 

if (norm_2(rEntityNormal))
rEntityNormal /= norm_2(rEntityNormal);

ProjectionValue = inner_prod(rEntityNormal, rNormal);

MeanCosinus += ProjectionValue;
}
}

if (TotalFaces != 0)
MeanCosinus *= (1.0 / TotalFaces);

if (MeanCosinus <= 0.3)
{
MeanCosinus = 0.8;
}

if (MeanCosinus > 3)
std::cout << " WARNING WRONG MeanCosinus " << MeanCosinus << std::endl;

if (MeanCosinus != 0 && MeanCosinus > 1e-3)
{ 
MeanCosinus = 1.0 / MeanCosinus;
rNormal *= MeanCosinus; 
}
else
{
std::cout << " Mean Cosinus not consistent " << MeanCosinus << " [" << (iNode)->Id() << "] rNormal " << rNormal[0] << " " << rNormal[1] << " " << rNormal[2] << std::endl;
}
}

rShrinkFactor = norm_2(rNormal);

if (rShrinkFactor != 0)
{
if (mEchoLevel > 1)
std::cout << "[Id " << (iNode)->Id() << " shrink_factor " << rShrinkFactor << " Normal " << rNormal[0] << " " << rNormal[1] << " " << rNormal[2] << " MeanCosinus " << MeanCosinus << "] shrink " << std::endl;
rNormal /= rShrinkFactor;
}
else
{

if (mEchoLevel > 1)
std::cout << "[Id " << (iNode)->Id() << " Normal " << rNormal[0] << " " << rNormal[1] << " " << rNormal[2] << " MeanCosinus " << MeanCosinus << "] no shrink " << std::endl;

noalias(rNormal) = ZeroVector(3);
rShrinkFactor = 1;

++not_assigned;
}
}

if (mEchoLevel > 0)
std::cout << "   [NORMALS SHRINKAGE (BOUNDARY NODES:" << NumberOfNodes << ") [SET:" << NumberOfNodes - not_assigned << " / NOT_SET:" << not_assigned << "] " << std::endl;

KRATOS_CATCH("")
}








































































































}; 




} 

#endif 
