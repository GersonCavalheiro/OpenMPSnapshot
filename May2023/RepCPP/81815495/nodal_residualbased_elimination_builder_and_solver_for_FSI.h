
#if !defined(KRATOS_NODAL_RESIDUAL_BASED_ELIMINATION_BUILDER_AND_SOLVER_FOR_FSI)
#define KRATOS_NODAL_RESIDUAL_BASED_ELIMINATION_BUILDER_AND_SOLVER_FOR_FSI


#include <set>
#ifdef _OPENMP
#include <omp.h>
#endif


#ifdef USE_GOOGLE_HASH
#include "sparsehash/dense_hash_set" 
#else
#include <unordered_set>
#endif


#include "utilities/timer.h"
#include "includes/define.h"
#include "includes/key_hash.h"
#include "solving_strategies/builder_and_solvers/builder_and_solver.h"
#include "includes/model_part.h"

#include "nodal_residualbased_elimination_builder_and_solver.h"

#include "pfem_fluid_dynamics_application_variables.h"

namespace Kratos
{







template <class TSparseSpace,
class TDenseSpace,  
class TLinearSolver 
>
class NodalResidualBasedEliminationBuilderAndSolverForFSI : public NodalResidualBasedEliminationBuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver>
{
public:
KRATOS_CLASS_POINTER_DEFINITION(NodalResidualBasedEliminationBuilderAndSolverForFSI);

typedef NodalResidualBasedEliminationBuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver> BaseType;

typedef typename BaseType::TSchemeType TSchemeType;

typedef typename BaseType::TDataType TDataType;

typedef typename BaseType::DofsArrayType DofsArrayType;

typedef typename BaseType::TSystemMatrixType TSystemMatrixType;

typedef typename BaseType::TSystemVectorType TSystemVectorType;

typedef typename BaseType::LocalSystemVectorType LocalSystemVectorType;

typedef typename BaseType::LocalSystemMatrixType LocalSystemMatrixType;

typedef typename BaseType::TSystemMatrixPointerType TSystemMatrixPointerType;
typedef typename BaseType::TSystemVectorPointerType TSystemVectorPointerType;

typedef Node NodeType;

typedef typename BaseType::NodesArrayType NodesArrayType;
typedef typename BaseType::ElementsArrayType ElementsArrayType;
typedef typename BaseType::ConditionsArrayType ConditionsArrayType;

typedef typename BaseType::ElementsContainerType ElementsContainerType;

typedef Vector VectorType;
typedef GlobalPointersVector<Node> NodeWeakPtrVectorType;



NodalResidualBasedEliminationBuilderAndSolverForFSI(
typename TLinearSolver::Pointer pNewLinearSystemSolver)
: NodalResidualBasedEliminationBuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver>(pNewLinearSystemSolver)
{
}


~NodalResidualBasedEliminationBuilderAndSolverForFSI() override
{
}



void SetMaterialPropertiesToSolid(
ModelPart::NodeIterator itNode,
double &density,
double &deviatoricCoeff,
double &volumetricCoeff,
double timeInterval,
double nodalVolume)
{
density = itNode->FastGetSolutionStepValue(SOLID_DENSITY);

double youngModulus = itNode->FastGetSolutionStepValue(YOUNG_MODULUS);
double poissonRatio = itNode->FastGetSolutionStepValue(POISSON_RATIO);

deviatoricCoeff = timeInterval * youngModulus / (1.0 + poissonRatio) * 0.5;
volumetricCoeff = timeInterval * poissonRatio * youngModulus / ((1.0 + poissonRatio) * (1.0 - 2.0 * poissonRatio)) + 2.0 * deviatoricCoeff / 3.0;
}

void BuildSolidNodally(
typename TSchemeType::Pointer pScheme,
ModelPart &rModelPart,
TSystemMatrixType &A,
TSystemVectorType &b,
double hybridCoeff)
{
KRATOS_TRY

KRATOS_ERROR_IF(!pScheme) << "No scheme provided!" << std::endl;

LocalSystemMatrixType solidLHS_Contribution = LocalSystemMatrixType(0, 0);
LocalSystemVectorType solidRHS_Contribution = LocalSystemVectorType(0);

Element::EquationIdVectorType solidEquationId;
ProcessInfo &CurrentProcessInfo = rModelPart.GetProcessInfo();

const unsigned int dimension = rModelPart.ElementsBegin()->GetGeometry().WorkingSpaceDimension();
const double timeInterval = CurrentProcessInfo[DELTA_TIME];
const double FourThirds = 4.0 / 3.0;
const double nTwoThirds = -2.0 / 3.0;

double theta = 1.0;
array_1d<double, 3> Acc(3, 0.0);

double dNdXi = 0;
double dNdYi = 0;
double dNdZi = 0;
double dNdXj = 0;
double dNdYj = 0;
double dNdZj = 0;
unsigned int firstRow = 0;
unsigned int firstCol = 0;

double density = 0;
double deviatoricCoeff = 0;
double volumetricCoeff = 0;

double dynamics = 1.0;

ModelPart::NodeIterator NodesBegin;
ModelPart::NodeIterator NodesEnd;
OpenMPUtils::PartitionedIterators(rModelPart.Nodes(), NodesBegin, NodesEnd);

double numNodesForExternalForce = 0;
double nodalExternalForce = 0;
bool belytsckoCase = false;
bool cooksMembraneCase = false;

if (cooksMembraneCase == true)
{
for (ModelPart::NodeIterator itNode = NodesBegin; itNode != NodesEnd; ++itNode)
{
double posX = itNode->X0();
if (posX > 47.999 && posX < 48.001)
{
numNodesForExternalForce += 1.0;
}
}
if (numNodesForExternalForce > 0)
{
nodalExternalForce = 1.0 / numNodesForExternalForce;
}
}

if (belytsckoCase == true)
{
for (ModelPart::NodeIterator itNode = NodesBegin; itNode != NodesEnd; ++itNode)
{
double posX = itNode->X0();
if (posX > 24.999 && posX < 25.001)
{
numNodesForExternalForce += 1.0;
}
}
if (numNodesForExternalForce > 0)
{
nodalExternalForce = 40.0 / numNodesForExternalForce;
}
}

for (ModelPart::NodeIterator itNode = NodesBegin; itNode != NodesEnd; ++itNode)
{

if (itNode->Is(SOLID))
{
NodeWeakPtrVectorType &neighb_nodes = itNode->GetValue(NEIGHBOUR_NODES);
Vector solidNodalSFDneighboursId = itNode->FastGetSolutionStepValue(SOLID_NODAL_SFD_NEIGHBOURS_ORDER);
const unsigned int neighSize = solidNodalSFDneighboursId.size();

const double nodalVolume = itNode->FastGetSolutionStepValue(SOLID_NODAL_VOLUME);

if (neighSize > 1 && nodalVolume > 0)
{

const unsigned int localSize = itNode->FastGetSolutionStepValue(SOLID_NODAL_SFD_NEIGHBOURS).size();

if (solidLHS_Contribution.size1() != localSize)
solidLHS_Contribution.resize(localSize, localSize, false); 

if (solidRHS_Contribution.size() != localSize)
solidRHS_Contribution.resize(localSize, false); 

if (solidEquationId.size() != localSize)
solidEquationId.resize(localSize, false);

noalias(solidLHS_Contribution) = ZeroMatrix(localSize, localSize);
noalias(solidRHS_Contribution) = ZeroVector(localSize);

this->SetMaterialPropertiesToSolid(itNode, density, deviatoricCoeff, volumetricCoeff, timeInterval, nodalVolume);

firstRow = 0;
firstCol = 0;

if (dimension == 2)
{
solidLHS_Contribution(0, 0) += nodalVolume * density * 2.0 * dynamics / timeInterval;
solidLHS_Contribution(1, 1) += nodalVolume * density * 2.0 * dynamics / timeInterval;

Acc = 2.0 * (itNode->FastGetSolutionStepValue(VELOCITY, 0) - itNode->FastGetSolutionStepValue(VELOCITY, 1)) / timeInterval - itNode->FastGetSolutionStepValue(ACCELERATION, 0);

solidRHS_Contribution[0] += -nodalVolume * density * Acc[0] * dynamics;
solidRHS_Contribution[1] += -nodalVolume * density * Acc[1] * dynamics;


array_1d<double, 3> &VolumeAcceleration = itNode->FastGetSolutionStepValue(VOLUME_ACCELERATION);


solidRHS_Contribution[0] += nodalVolume * density * VolumeAcceleration[0];
solidRHS_Contribution[1] += nodalVolume * density * VolumeAcceleration[1];


if (belytsckoCase == true)
{
if (itNode->X0() > 24.999 && itNode->X0() < 25.001)
{
solidRHS_Contribution[1] += nodalExternalForce;
}
}
if (cooksMembraneCase == true)
{
if (itNode->X0() > 47.999 && itNode->X0() < 48.001)
{
solidRHS_Contribution[1] += nodalExternalForce;
}
}

array_1d<double, 3> Sigma(3, 0.0);
Sigma = itNode->FastGetSolutionStepValue(SOLID_NODAL_CAUCHY_STRESS);

const unsigned int xDofPos = itNode->GetDofPosition(VELOCITY_X);
solidEquationId[0] = itNode->GetDof(VELOCITY_X, xDofPos).EquationId();
solidEquationId[1] = itNode->GetDof(VELOCITY_Y, xDofPos + 1).EquationId();

for (unsigned int i = 0; i < neighSize; i++)
{
dNdXi = itNode->FastGetSolutionStepValue(SOLID_NODAL_SFD_NEIGHBOURS)[firstCol];
dNdYi = itNode->FastGetSolutionStepValue(SOLID_NODAL_SFD_NEIGHBOURS)[firstCol + 1];

solidRHS_Contribution[firstCol] += -nodalVolume * (dNdXi * Sigma[0] + dNdYi * Sigma[2]) * hybridCoeff;
solidRHS_Contribution[firstCol + 1] += -nodalVolume * (dNdYi * Sigma[1] + dNdXi * Sigma[2]) * hybridCoeff;

for (unsigned int j = 0; j < neighSize; j++)
{
dNdXj = itNode->FastGetSolutionStepValue(SOLID_NODAL_SFD_NEIGHBOURS)[firstRow];
dNdYj = itNode->FastGetSolutionStepValue(SOLID_NODAL_SFD_NEIGHBOURS)[firstRow + 1];

solidLHS_Contribution(firstRow, firstCol) += nodalVolume * ((FourThirds * deviatoricCoeff + volumetricCoeff) * dNdXj * dNdXi + dNdYj * dNdYi * deviatoricCoeff) * theta * hybridCoeff;
solidLHS_Contribution(firstRow, firstCol + 1) += nodalVolume * ((nTwoThirds * deviatoricCoeff + volumetricCoeff) * dNdXj * dNdYi + dNdYj * dNdXi * deviatoricCoeff) * theta * hybridCoeff;

solidLHS_Contribution(firstRow + 1, firstCol) += nodalVolume * ((nTwoThirds * deviatoricCoeff + volumetricCoeff) * dNdYj * dNdXi + dNdXj * dNdYi * deviatoricCoeff) * theta * hybridCoeff;
solidLHS_Contribution(firstRow + 1, firstCol + 1) += nodalVolume * ((FourThirds * deviatoricCoeff + volumetricCoeff) * dNdYj * dNdYi + dNdXj * dNdXi * deviatoricCoeff) * theta * hybridCoeff;

firstRow += 2;
}

firstRow = 0;
firstCol += 2;

unsigned int indexNode = i + 1;
if (itNode->FastGetSolutionStepValue(INTERFACE_NODE) == true && indexNode < neighSize)
{
unsigned int other_neigh_nodes_id = solidNodalSFDneighboursId[indexNode];
for (unsigned int k = 0; k < neighb_nodes.size(); k++)
{
unsigned int neigh_nodes_id = neighb_nodes[k].Id();

if (neigh_nodes_id == other_neigh_nodes_id)
{
solidEquationId[firstCol] = neighb_nodes[k].GetDof(VELOCITY_X, xDofPos).EquationId();
solidEquationId[firstCol + 1] = neighb_nodes[k].GetDof(VELOCITY_Y, xDofPos + 1).EquationId();
break;
}
}
}
else if (i < neighb_nodes.size())
{
solidEquationId[firstCol] = neighb_nodes[i].GetDof(VELOCITY_X, xDofPos).EquationId();
solidEquationId[firstCol + 1] = neighb_nodes[i].GetDof(VELOCITY_Y, xDofPos + 1).EquationId();
}
}
}
else if (dimension == 3)
{
solidLHS_Contribution(0, 0) += nodalVolume * density * 2.0 / timeInterval;
solidLHS_Contribution(1, 1) += nodalVolume * density * 2.0 / timeInterval;
solidLHS_Contribution(2, 2) += nodalVolume * density * 2.0 / timeInterval;

Acc = 2.0 * (itNode->FastGetSolutionStepValue(VELOCITY, 0) - itNode->FastGetSolutionStepValue(VELOCITY, 1)) / timeInterval - itNode->FastGetSolutionStepValue(ACCELERATION, 0);

solidRHS_Contribution[0] += -nodalVolume * density * Acc[0];
solidRHS_Contribution[1] += -nodalVolume * density * Acc[1];
solidRHS_Contribution[2] += -nodalVolume * density * Acc[2];


array_1d<double, 3> &VolumeAcceleration = itNode->FastGetSolutionStepValue(VOLUME_ACCELERATION);

solidRHS_Contribution[0] += nodalVolume * density * VolumeAcceleration[0];
solidRHS_Contribution[1] += nodalVolume * density * VolumeAcceleration[1];
solidRHS_Contribution[2] += nodalVolume * density * VolumeAcceleration[2];



array_1d<double, 6> Sigma(6, 0.0);
Sigma = itNode->FastGetSolutionStepValue(SOLID_NODAL_CAUCHY_STRESS);

const unsigned int xDofPos = itNode->GetDofPosition(VELOCITY_X);
solidEquationId[0] = itNode->GetDof(VELOCITY_X, xDofPos).EquationId();
solidEquationId[1] = itNode->GetDof(VELOCITY_Y, xDofPos + 1).EquationId();
solidEquationId[2] = itNode->GetDof(VELOCITY_Z, xDofPos + 2).EquationId();

for (unsigned int i = 0; i < neighSize; i++)
{
dNdXi = itNode->FastGetSolutionStepValue(SOLID_NODAL_SFD_NEIGHBOURS)[firstCol];
dNdYi = itNode->FastGetSolutionStepValue(SOLID_NODAL_SFD_NEIGHBOURS)[firstCol + 1];
dNdZi = itNode->FastGetSolutionStepValue(SOLID_NODAL_SFD_NEIGHBOURS)[firstCol + 2];

solidRHS_Contribution[firstCol] += -nodalVolume * (dNdXi * Sigma[0] + dNdYi * Sigma[3] + dNdZi * Sigma[4]);
solidRHS_Contribution[firstCol + 1] += -nodalVolume * (dNdYi * Sigma[1] + dNdXi * Sigma[3] + dNdZi * Sigma[5]);
solidRHS_Contribution[firstCol + 2] += -nodalVolume * (dNdZi * Sigma[2] + dNdXi * Sigma[4] + dNdYi * Sigma[5]);

for (unsigned int j = 0; j < neighSize; j++)
{

dNdXj = itNode->FastGetSolutionStepValue(SOLID_NODAL_SFD_NEIGHBOURS)[firstRow];
dNdYj = itNode->FastGetSolutionStepValue(SOLID_NODAL_SFD_NEIGHBOURS)[firstRow + 1];
dNdZj = itNode->FastGetSolutionStepValue(SOLID_NODAL_SFD_NEIGHBOURS)[firstRow + 2];

solidLHS_Contribution(firstRow, firstCol) += nodalVolume * ((FourThirds * deviatoricCoeff + volumetricCoeff) * dNdXj * dNdXi + (dNdYj * dNdYi + dNdZj * dNdZi) * deviatoricCoeff) * theta;
solidLHS_Contribution(firstRow, firstCol + 1) += nodalVolume * ((nTwoThirds * deviatoricCoeff + volumetricCoeff) * dNdXj * dNdYi + dNdYj * dNdXi * deviatoricCoeff) * theta;
solidLHS_Contribution(firstRow, firstCol + 2) += nodalVolume * ((nTwoThirds * deviatoricCoeff + volumetricCoeff) * dNdXj * dNdZi + dNdZj * dNdXi * deviatoricCoeff) * theta;

solidLHS_Contribution(firstRow + 1, firstCol) += nodalVolume * ((nTwoThirds * deviatoricCoeff + volumetricCoeff) * dNdYj * dNdXi + dNdXj * dNdYi * deviatoricCoeff) * theta;
solidLHS_Contribution(firstRow + 1, firstCol + 1) += nodalVolume * ((FourThirds * deviatoricCoeff + volumetricCoeff) * dNdYj * dNdYi + (dNdXj * dNdXi + dNdZj * dNdZi) * deviatoricCoeff) * theta;
solidLHS_Contribution(firstRow + 1, firstCol + 2) += nodalVolume * ((nTwoThirds * deviatoricCoeff + volumetricCoeff) * dNdYj * dNdZi + dNdZj * dNdYi * deviatoricCoeff) * theta;

solidLHS_Contribution(firstRow + 2, firstCol) += nodalVolume * ((nTwoThirds * deviatoricCoeff + volumetricCoeff) * dNdZj * dNdXi + dNdXj * dNdZi * deviatoricCoeff) * theta;
solidLHS_Contribution(firstRow + 2, firstCol + 1) += nodalVolume * ((nTwoThirds * deviatoricCoeff + volumetricCoeff) * dNdZj * dNdYi + dNdYj * dNdZi * deviatoricCoeff) * theta;
solidLHS_Contribution(firstRow + 2, firstCol + 2) += nodalVolume * ((FourThirds * deviatoricCoeff + volumetricCoeff) * dNdZj * dNdZi + (dNdXj * dNdXi + dNdYj * dNdYi) * deviatoricCoeff) * theta;

firstRow += 3;
}

firstRow = 0;
firstCol += 3;

unsigned int indexNode = i + 1;
if (itNode->FastGetSolutionStepValue(INTERFACE_NODE) == true && indexNode < neighSize)
{
unsigned int other_neigh_nodes_id = solidNodalSFDneighboursId[indexNode];
for (unsigned int k = 0; k < neighb_nodes.size(); k++)
{
unsigned int neigh_nodes_id = neighb_nodes[k].Id();

if (neigh_nodes_id == other_neigh_nodes_id)
{
solidEquationId[firstCol] = neighb_nodes[k].GetDof(VELOCITY_X, xDofPos).EquationId();
solidEquationId[firstCol + 1] = neighb_nodes[k].GetDof(VELOCITY_Y, xDofPos + 1).EquationId();
solidEquationId[firstCol + 2] = neighb_nodes[k].GetDof(VELOCITY_Z, xDofPos + 2).EquationId();
break;
}
}
}
else if (i < neighb_nodes.size())
{
solidEquationId[firstCol] = neighb_nodes[i].GetDof(VELOCITY_X, xDofPos).EquationId();
solidEquationId[firstCol + 1] = neighb_nodes[i].GetDof(VELOCITY_Y, xDofPos + 1).EquationId();
solidEquationId[firstCol + 2] = neighb_nodes[i].GetDof(VELOCITY_Z, xDofPos + 2).EquationId();
}
}
}

#ifdef _OPENMP
Assemble(A, b, solidLHS_Contribution, solidRHS_Contribution, solidEquationId, mlock_array);
#else
Assemble(A, b, solidLHS_Contribution, solidRHS_Contribution, solidEquationId);
#endif
}
}
}


KRATOS_CATCH("")
}

void BuildFluidNodally(
typename TSchemeType::Pointer pScheme,
ModelPart &rModelPart,
TSystemMatrixType &A,
TSystemVectorType &b)
{
KRATOS_TRY

KRATOS_ERROR_IF(!pScheme) << "No scheme provided!" << std::endl;



LocalSystemMatrixType LHS_Contribution = LocalSystemMatrixType(0, 0);
LocalSystemVectorType RHS_Contribution = LocalSystemVectorType(0);

Element::EquationIdVectorType EquationId;
ProcessInfo &CurrentProcessInfo = rModelPart.GetProcessInfo();

const unsigned int dimension = rModelPart.ElementsBegin()->GetGeometry().WorkingSpaceDimension();
const double timeInterval = CurrentProcessInfo[DELTA_TIME];
const double FourThirds = 4.0 / 3.0;
const double nTwoThirds = -2.0 / 3.0;

double theta = 0.5;
array_1d<double, 3> Acc(3, 0.0);
double pressure = 0;
double dNdXi = 0;
double dNdYi = 0;
double dNdZi = 0;
double dNdXj = 0;
double dNdYj = 0;
double dNdZj = 0;
unsigned int firstRow = 0;
unsigned int firstCol = 0;

double density = 0;
double deviatoricCoeff = 0;
double volumetricCoeff = 0;


ModelPart::NodeIterator NodesBegin;
ModelPart::NodeIterator NodesEnd;
OpenMPUtils::PartitionedIterators(rModelPart.Nodes(), NodesBegin, NodesEnd);

for (ModelPart::NodeIterator itNode = NodesBegin; itNode != NodesEnd; ++itNode)
{

if ((itNode->Is(FLUID) && itNode->IsNot(SOLID)) || itNode->FastGetSolutionStepValue(INTERFACE_NODE) == true)
{

NodeWeakPtrVectorType &neighb_nodes = itNode->GetValue(NEIGHBOUR_NODES);
Vector nodalSFDneighboursId = itNode->FastGetSolutionStepValue(NODAL_SFD_NEIGHBOURS_ORDER);
const unsigned int neighSize = nodalSFDneighboursId.size();
const double nodalVolume = itNode->FastGetSolutionStepValue(NODAL_VOLUME);

if (neighSize > 1 && nodalVolume > 0)
{

const unsigned int localSize = itNode->FastGetSolutionStepValue(NODAL_SFD_NEIGHBOURS).size();

if (LHS_Contribution.size1() != localSize)
LHS_Contribution.resize(localSize, localSize, false); 

if (RHS_Contribution.size() != localSize)
RHS_Contribution.resize(localSize, false); 

if (EquationId.size() != localSize)
EquationId.resize(localSize, false);

noalias(LHS_Contribution) = ZeroMatrix(localSize, localSize);
noalias(RHS_Contribution) = ZeroVector(localSize);

this->SetMaterialPropertiesToFluid(itNode, density, deviatoricCoeff, volumetricCoeff, timeInterval, nodalVolume);

firstRow = 0;
firstCol = 0;

if (dimension == 2)
{
LHS_Contribution(0, 0) += nodalVolume * density * 2.0 / timeInterval;
LHS_Contribution(1, 1) += nodalVolume * density * 2.0 / timeInterval;

Acc = 2.0 * (itNode->FastGetSolutionStepValue(VELOCITY, 0) - itNode->FastGetSolutionStepValue(VELOCITY, 1)) / timeInterval -
itNode->FastGetSolutionStepValue(ACCELERATION, 0);

RHS_Contribution[0] += -nodalVolume * density * Acc[0];
RHS_Contribution[1] += -nodalVolume * density * Acc[1];


array_1d<double, 3> &VolumeAcceleration = itNode->FastGetSolutionStepValue(VOLUME_ACCELERATION);


RHS_Contribution[0] += nodalVolume * density * VolumeAcceleration[0];
RHS_Contribution[1] += nodalVolume * density * VolumeAcceleration[1];

array_1d<double, 3> Sigma(3, 0.0);
Sigma = itNode->FastGetSolutionStepValue(NODAL_CAUCHY_STRESS);

if (itNode->IsNot(SOLID) || itNode->FastGetSolutionStepValue(INTERFACE_NODE) == true)
{
pressure = itNode->FastGetSolutionStepValue(PRESSURE, 0) * theta + itNode->FastGetSolutionStepValue(PRESSURE, 1) * (1 - theta);
Sigma[0] = itNode->FastGetSolutionStepValue(NODAL_DEVIATORIC_CAUCHY_STRESS)[0] + pressure;
Sigma[1] = itNode->FastGetSolutionStepValue(NODAL_DEVIATORIC_CAUCHY_STRESS)[1] + pressure;
}

const unsigned int xDofPos = itNode->GetDofPosition(VELOCITY_X);
EquationId[0] = itNode->GetDof(VELOCITY_X, xDofPos).EquationId();
EquationId[1] = itNode->GetDof(VELOCITY_Y, xDofPos + 1).EquationId();

for (unsigned int i = 0; i < neighSize; i++)
{
dNdXi = itNode->FastGetSolutionStepValue(NODAL_SFD_NEIGHBOURS)[firstCol];
dNdYi = itNode->FastGetSolutionStepValue(NODAL_SFD_NEIGHBOURS)[firstCol + 1];

RHS_Contribution[firstCol] += -nodalVolume * (dNdXi * Sigma[0] + dNdYi * Sigma[2]);
RHS_Contribution[firstCol + 1] += -nodalVolume * (dNdYi * Sigma[1] + dNdXi * Sigma[2]);

for (unsigned int j = 0; j < neighSize; j++)
{
dNdXj = itNode->FastGetSolutionStepValue(NODAL_SFD_NEIGHBOURS)[firstRow];
dNdYj = itNode->FastGetSolutionStepValue(NODAL_SFD_NEIGHBOURS)[firstRow + 1];

LHS_Contribution(firstRow, firstCol) += nodalVolume * ((FourThirds * deviatoricCoeff + volumetricCoeff) * dNdXj * dNdXi + dNdYj * dNdYi * deviatoricCoeff) * theta;
LHS_Contribution(firstRow, firstCol + 1) += nodalVolume * ((nTwoThirds * deviatoricCoeff + volumetricCoeff) * dNdXj * dNdYi + dNdYj * dNdXi * deviatoricCoeff) * theta;

LHS_Contribution(firstRow + 1, firstCol) += nodalVolume * ((nTwoThirds * deviatoricCoeff + volumetricCoeff) * dNdYj * dNdXi + dNdXj * dNdYi * deviatoricCoeff) * theta;
LHS_Contribution(firstRow + 1, firstCol + 1) += nodalVolume * ((FourThirds * deviatoricCoeff + volumetricCoeff) * dNdYj * dNdYi + dNdXj * dNdXi * deviatoricCoeff) * theta;

firstRow += 2;
}

firstRow = 0;
firstCol += 2;

unsigned int indexNode = i + 1;
if (itNode->FastGetSolutionStepValue(INTERFACE_NODE) == true && indexNode < neighSize)
{
unsigned int other_neigh_nodes_id = nodalSFDneighboursId[indexNode];
for (unsigned int k = 0; k < neighb_nodes.size(); k++)
{
unsigned int neigh_nodes_id = neighb_nodes[k].Id();

if (neigh_nodes_id == other_neigh_nodes_id)
{
EquationId[firstCol] = neighb_nodes[k].GetDof(VELOCITY_X, xDofPos).EquationId();
EquationId[firstCol + 1] = neighb_nodes[k].GetDof(VELOCITY_Y, xDofPos + 1).EquationId();
break;
}
}
}
else if (i < neighb_nodes.size())
{
EquationId[firstCol] = neighb_nodes[i].GetDof(VELOCITY_X, xDofPos).EquationId();
EquationId[firstCol + 1] = neighb_nodes[i].GetDof(VELOCITY_Y, xDofPos + 1).EquationId();
}
}
}
else if (dimension == 3)
{
LHS_Contribution(0, 0) += nodalVolume * density * 2.0 / timeInterval;
LHS_Contribution(1, 1) += nodalVolume * density * 2.0 / timeInterval;
LHS_Contribution(2, 2) += nodalVolume * density * 2.0 / timeInterval;

Acc = 2.0 * (itNode->FastGetSolutionStepValue(VELOCITY, 0) - itNode->FastGetSolutionStepValue(VELOCITY, 1)) / timeInterval -
itNode->FastGetSolutionStepValue(ACCELERATION, 0);

RHS_Contribution[0] += -nodalVolume * density * Acc[0];
RHS_Contribution[1] += -nodalVolume * density * Acc[1];
RHS_Contribution[2] += -nodalVolume * density * Acc[2];


array_1d<double, 3> &VolumeAcceleration = itNode->FastGetSolutionStepValue(VOLUME_ACCELERATION);

RHS_Contribution[0] += nodalVolume * density * VolumeAcceleration[0];
RHS_Contribution[1] += nodalVolume * density * VolumeAcceleration[1];
RHS_Contribution[2] += nodalVolume * density * VolumeAcceleration[2];


array_1d<double, 6> Sigma(6, 0.0);
Sigma = itNode->FastGetSolutionStepValue(NODAL_CAUCHY_STRESS);

if (itNode->IsNot(SOLID) || itNode->FastGetSolutionStepValue(INTERFACE_NODE) == true)
{
pressure = itNode->FastGetSolutionStepValue(PRESSURE, 0) * theta + itNode->FastGetSolutionStepValue(PRESSURE, 1) * (1 - theta);
Sigma[0] = itNode->FastGetSolutionStepValue(NODAL_DEVIATORIC_CAUCHY_STRESS)[0] + pressure;
Sigma[1] = itNode->FastGetSolutionStepValue(NODAL_DEVIATORIC_CAUCHY_STRESS)[1] + pressure;
Sigma[2] = itNode->FastGetSolutionStepValue(NODAL_DEVIATORIC_CAUCHY_STRESS)[2] + pressure;
}

const unsigned int xDofPos = itNode->GetDofPosition(VELOCITY_X);
EquationId[0] = itNode->GetDof(VELOCITY_X, xDofPos).EquationId();
EquationId[1] = itNode->GetDof(VELOCITY_Y, xDofPos + 1).EquationId();
EquationId[2] = itNode->GetDof(VELOCITY_Z, xDofPos + 2).EquationId();

for (unsigned int i = 0; i < neighSize; i++)
{
dNdXi = itNode->FastGetSolutionStepValue(NODAL_SFD_NEIGHBOURS)[firstCol];
dNdYi = itNode->FastGetSolutionStepValue(NODAL_SFD_NEIGHBOURS)[firstCol + 1];
dNdZi = itNode->FastGetSolutionStepValue(NODAL_SFD_NEIGHBOURS)[firstCol + 2];

RHS_Contribution[firstCol] += -nodalVolume * (dNdXi * Sigma[0] + dNdYi * Sigma[3] + dNdZi * Sigma[4]);
RHS_Contribution[firstCol + 1] += -nodalVolume * (dNdYi * Sigma[1] + dNdXi * Sigma[3] + dNdZi * Sigma[5]);
RHS_Contribution[firstCol + 2] += -nodalVolume * (dNdZi * Sigma[2] + dNdXi * Sigma[4] + dNdYi * Sigma[5]);

for (unsigned int j = 0; j < neighSize; j++)
{

dNdXj = itNode->FastGetSolutionStepValue(NODAL_SFD_NEIGHBOURS)[firstRow];
dNdYj = itNode->FastGetSolutionStepValue(NODAL_SFD_NEIGHBOURS)[firstRow + 1];
dNdZj = itNode->FastGetSolutionStepValue(NODAL_SFD_NEIGHBOURS)[firstRow + 2];

LHS_Contribution(firstRow, firstCol) += nodalVolume * ((FourThirds * deviatoricCoeff + volumetricCoeff) * dNdXj * dNdXi + (dNdYj * dNdYi + dNdZj * dNdZi) * deviatoricCoeff) * theta;
LHS_Contribution(firstRow, firstCol + 1) += nodalVolume * ((nTwoThirds * deviatoricCoeff + volumetricCoeff) * dNdXj * dNdYi + dNdYj * dNdXi * deviatoricCoeff) * theta;
LHS_Contribution(firstRow, firstCol + 2) += nodalVolume * ((nTwoThirds * deviatoricCoeff + volumetricCoeff) * dNdXj * dNdZi + dNdZj * dNdXi * deviatoricCoeff) * theta;

LHS_Contribution(firstRow + 1, firstCol) += nodalVolume * ((nTwoThirds * deviatoricCoeff + volumetricCoeff) * dNdYj * dNdXi + dNdXj * dNdYi * deviatoricCoeff) * theta;
LHS_Contribution(firstRow + 1, firstCol + 1) += nodalVolume * ((FourThirds * deviatoricCoeff + volumetricCoeff) * dNdYj * dNdYi + (dNdXj * dNdXi + dNdZj * dNdZi) * deviatoricCoeff) * theta;
LHS_Contribution(firstRow + 1, firstCol + 2) += nodalVolume * ((nTwoThirds * deviatoricCoeff + volumetricCoeff) * dNdYj * dNdZi + dNdZj * dNdYi * deviatoricCoeff) * theta;

LHS_Contribution(firstRow + 2, firstCol) += nodalVolume * ((nTwoThirds * deviatoricCoeff + volumetricCoeff) * dNdZj * dNdXi + dNdXj * dNdZi * deviatoricCoeff) * theta;
LHS_Contribution(firstRow + 2, firstCol + 1) += nodalVolume * ((nTwoThirds * deviatoricCoeff + volumetricCoeff) * dNdZj * dNdYi + dNdYj * dNdZi * deviatoricCoeff) * theta;
LHS_Contribution(firstRow + 2, firstCol + 2) += nodalVolume * ((FourThirds * deviatoricCoeff + volumetricCoeff) * dNdZj * dNdZi + (dNdXj * dNdXi + dNdYj * dNdYi) * deviatoricCoeff) * theta;

firstRow += 3;
}

firstRow = 0;
firstCol += 3;

unsigned int indexNode = i + 1;
if (itNode->FastGetSolutionStepValue(INTERFACE_NODE) == true && indexNode < neighSize)
{
unsigned int other_neigh_nodes_id = nodalSFDneighboursId[indexNode];
for (unsigned int k = 0; k < neighb_nodes.size(); k++)
{
unsigned int neigh_nodes_id = neighb_nodes[k].Id();

if (neigh_nodes_id == other_neigh_nodes_id)
{
EquationId[firstCol] = neighb_nodes[k].GetDof(VELOCITY_X, xDofPos).EquationId();
EquationId[firstCol + 1] = neighb_nodes[k].GetDof(VELOCITY_Y, xDofPos + 1).EquationId();
EquationId[firstCol + 2] = neighb_nodes[k].GetDof(VELOCITY_Z, xDofPos + 2).EquationId();
break;
}
}
}
else if (i < neighb_nodes.size())
{
EquationId[firstCol] = neighb_nodes[i].GetDof(VELOCITY_X, xDofPos).EquationId();
EquationId[firstCol + 1] = neighb_nodes[i].GetDof(VELOCITY_Y, xDofPos + 1).EquationId();
EquationId[firstCol + 2] = neighb_nodes[i].GetDof(VELOCITY_Z, xDofPos + 2).EquationId();
}
}
}

#ifdef _OPENMP
Assemble(A, b, LHS_Contribution, RHS_Contribution, EquationId, mlock_array);
#else
Assemble(A, b, LHS_Contribution, RHS_Contribution, EquationId);
#endif
}
}
}


KRATOS_CATCH("")
}


void BuildAndSolve(
typename TSchemeType::Pointer pScheme,
ModelPart &rModelPart,
TSystemMatrixType &A,
TSystemVectorType &Dx,
TSystemVectorType &b) override
{
KRATOS_TRY

Timer::Start("Build");


double hybridCoeff = 1.0; 

BuildSolidNodally(pScheme, rModelPart, A, b, hybridCoeff);

if (hybridCoeff < 0.99999999)
{
BuildElementally(pScheme, rModelPart, A, b);
}

BuildFluidNodally(pScheme, rModelPart, A, b);


Timer::Stop("Build");


this->ApplyDirichletConditions(pScheme, rModelPart, A, Dx, b);

KRATOS_INFO_IF("ResidualBasedBlockBuilderAndSolver", (this->GetEchoLevel() == 3)) << "Before the solution of the system"
<< "\nSystem Matrix = " << A << "\nUnknowns vector = " << Dx << "\nRHS vector = " << b << std::endl;

this->SystemSolveWithPhysics(A, Dx, b, rModelPart);

KRATOS_INFO_IF("ResidualBasedBlockBuilderAndSolver", (this->GetEchoLevel() == 3)) << "After the solution of the system"
<< "\nSystem Matrix = " << A << "\nUnknowns vector = " << Dx << "\nRHS vector = " << b << std::endl;

KRATOS_CATCH("")
}

void BuildElementally(
typename TSchemeType::Pointer pScheme,
ModelPart &rModelPart,
TSystemMatrixType &rA,
TSystemVectorType &rb)
{
KRATOS_TRY

KRATOS_ERROR_IF(!pScheme) << "No scheme provided!" << std::endl;

const int nelements = static_cast<int>(rModelPart.Elements().size());

const int nconditions = static_cast<int>(rModelPart.Conditions().size());

const ProcessInfo &CurrentProcessInfo = rModelPart.GetProcessInfo();
ModelPart::ElementsContainerType::iterator el_begin = rModelPart.ElementsBegin();
ModelPart::ConditionsContainerType::iterator cond_begin = rModelPart.ConditionsBegin();

LocalSystemMatrixType LHS_Contribution = LocalSystemMatrixType(0, 0);
LocalSystemVectorType RHS_Contribution = LocalSystemVectorType(0);

Element::EquationIdVectorType EquationId;

#pragma omp parallel firstprivate(nelements, nconditions, LHS_Contribution, RHS_Contribution, EquationId)
{
#pragma omp for schedule(guided, 512) nowait
for (int k = 0; k < nelements; k++)
{
ModelPart::ElementsContainerType::iterator it = el_begin + k;

bool element_is_active = true;
if ((it)->IsDefined(ACTIVE))
element_is_active = (it)->Is(ACTIVE);

if (element_is_active)
{
pScheme->CalculateSystemContributions(*it, LHS_Contribution, RHS_Contribution, EquationId, CurrentProcessInfo);

#ifdef USE_LOCKS_IN_ASSEMBLY
AssembleElementally(rA, rb, LHS_Contribution, RHS_Contribution, EquationId, mLockArray);
#else
AssembleElementally(rA, rb, LHS_Contribution, RHS_Contribution, EquationId);
#endif
}
}

#pragma omp for schedule(guided, 512)
for (int k = 0; k < nconditions; k++)
{
ModelPart::ConditionsContainerType::iterator it = cond_begin + k;

bool condition_is_active = true;
if ((it)->IsDefined(ACTIVE))
condition_is_active = (it)->Is(ACTIVE);

if (condition_is_active)
{
pScheme->CalculateSystemContributions(*it, LHS_Contribution, RHS_Contribution, EquationId, CurrentProcessInfo);

#ifdef USE_LOCKS_IN_ASSEMBLY
AssembleElementally(rA, rb, LHS_Contribution, RHS_Contribution, EquationId, mLockArray);
#else
AssembleElementally(rA, rb, LHS_Contribution, RHS_Contribution, EquationId);
#endif
}
}
}
KRATOS_INFO_IF("ResidualBasedEliminationBuilderAndSolver", this->GetEchoLevel() > 2 && rModelPart.GetCommunicator().MyPID() == 0) << "Finished building" << std::endl;

KRATOS_CATCH("")
}

void AssembleElementally(
TSystemMatrixType &rA,
TSystemVectorType &rb,
const LocalSystemMatrixType &rLHSContribution,
const LocalSystemVectorType &rRHSContribution,
const Element::EquationIdVectorType &rEquationId
#ifdef USE_LOCKS_IN_ASSEMBLY
,
std::vector<omp_lock_t> &rLockArray
#endif
)
{
unsigned int local_size = rLHSContribution.size1();

for (unsigned int i_local = 0; i_local < local_size; i_local++)
{
unsigned int i_global = rEquationId[i_local];

if (i_global < BaseType::mEquationSystemSize)
{
#ifdef USE_LOCKS_IN_ASSEMBLY
omp_set_lock(&rLockArray[i_global]);
b[i_global] += rRHSContribution(i_local);
#else
double &r_a = rb[i_global];
const double &v_a = rRHSContribution(i_local);
#pragma omp atomic
r_a += v_a;
#endif
AssembleRowContributionFreeDofs(rA, rLHSContribution, i_global, i_local, rEquationId);

#ifdef USE_LOCKS_IN_ASSEMBLY
omp_unset_lock(&rLockArray[i_global]);
#endif
}
}
}


void SetUpDofSet(
typename TSchemeType::Pointer pScheme,
ModelPart &rModelPart) override
{
KRATOS_TRY;

KRATOS_INFO_IF("NodalResidualBasedEliminationBuilderAndSolverForFSI", this->GetEchoLevel() > 1 && rModelPart.GetCommunicator().MyPID() == 0) << "Setting up the dofs" << std::endl;

ElementsArrayType &pElements = rModelPart.Elements();
const int nelements = static_cast<int>(pElements.size());

Element::DofsVectorType ElementalDofList;

ProcessInfo &CurrentProcessInfo = rModelPart.GetProcessInfo();

unsigned int nthreads = ParallelUtilities::GetNumThreads();


#ifdef USE_GOOGLE_HASH
typedef google::dense_hash_set<NodeType::DofType::Pointer, DofPointerHasher> set_type;
#else
typedef std::unordered_set<NodeType::DofType::Pointer, DofPointerHasher> set_type;
#endif

std::vector<set_type> dofs_aux_list(nthreads);

for (int i = 0; i < static_cast<int>(nthreads); i++)
{
#ifdef USE_GOOGLE_HASH
dofs_aux_list[i].set_empty_key(NodeType::DofType::Pointer());
#else
dofs_aux_list[i].reserve(nelements);
#endif
}

for (int i = 0; i < static_cast<int>(nelements); ++i)
{
auto it_elem = pElements.begin() + i;
const IndexType this_thread_id = OpenMPUtils::ThisThread();

pScheme->GetDofList(*it_elem, ElementalDofList, CurrentProcessInfo);

dofs_aux_list[this_thread_id].insert(ElementalDofList.begin(), ElementalDofList.end());
}

ConditionsArrayType &pConditions = rModelPart.Conditions();
const int nconditions = static_cast<int>(pConditions.size());
#pragma omp parallel for firstprivate(nconditions, ElementalDofList)
for (int i = 0; i < nconditions; ++i)
{
auto it_cond = pConditions.begin() + i;
const IndexType this_thread_id = OpenMPUtils::ThisThread();

pScheme->GetDofList(*it_cond, ElementalDofList, CurrentProcessInfo);
dofs_aux_list[this_thread_id].insert(ElementalDofList.begin(), ElementalDofList.end());
}

unsigned int old_max = nthreads;
unsigned int new_max = ceil(0.5 * static_cast<double>(old_max));
while (new_max >= 1 && new_max != old_max)
{

#pragma omp parallel for
for (int i = 0; i < static_cast<int>(new_max); i++)
{
if (i + new_max < old_max)
{
dofs_aux_list[i].insert(dofs_aux_list[i + new_max].begin(), dofs_aux_list[i + new_max].end());
dofs_aux_list[i + new_max].clear();
}
}

old_max = new_max;
new_max = ceil(0.5 * static_cast<double>(old_max));
}

DofsArrayType Doftemp;
BaseType::mDofSet = DofsArrayType();

Doftemp.reserve(dofs_aux_list[0].size());
for (auto it = dofs_aux_list[0].begin(); it != dofs_aux_list[0].end(); it++)
{
Doftemp.push_back(*it);
}
Doftemp.Sort();

BaseType::mDofSet = Doftemp;

KRATOS_ERROR_IF(BaseType::mDofSet.size() == 0) << "No degrees of freedom!" << std::endl;

BaseType::mDofSetIsInitialized = true;

KRATOS_INFO_IF("NodalResidualBasedEliminationBuilderAndSolverForFSI", this->GetEchoLevel() > 2 && rModelPart.GetCommunicator().MyPID() == 0) << "Finished setting up the dofs" << std::endl;

#ifdef _OPENMP
if (mlock_array.size() != 0)
{
for (int i = 0; i < static_cast<int>(mlock_array.size()); i++)
omp_destroy_lock(&mlock_array[i]);
}

mlock_array.resize(BaseType::mDofSet.size());

for (int i = 0; i < static_cast<int>(mlock_array.size()); i++)
omp_init_lock(&mlock_array[i]);
#endif

#ifdef KRATOS_DEBUG
if (BaseType::GetCalculateReactionsFlag())
{
for (auto dof_iterator = BaseType::mDofSet.begin(); dof_iterator != BaseType::mDofSet.end(); ++dof_iterator)
{
KRATOS_ERROR_IF_NOT(dof_iterator->HasReaction()) << "Reaction variable not set for the following : " << std::endl
<< "Node : " << dof_iterator->Id() << std::endl
<< "Dof : " << (*dof_iterator) << std::endl
<< "Not possible to calculate reactions." << std::endl;
}
}
#endif

KRATOS_CATCH("");
}

/




} 

#endif 
