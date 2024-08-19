
#if !defined(KRATOS_NODAL_RESIDUAL_BASED_ELIMINATION_BUILDER_AND_SOLVER_CONTINUITY)
#define KRATOS_NODAL_RESIDUAL_BASED_ELIMINATION_BUILDER_AND_SOLVER_CONTINUITY


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

#include "pfem_fluid_dynamics_application_variables.h"

namespace Kratos
{







template <class TSparseSpace,
class TDenseSpace,  
class TLinearSolver 
>
class NodalResidualBasedEliminationBuilderAndSolverContinuity
: public BuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver>
{
public:
KRATOS_CLASS_POINTER_DEFINITION(NodalResidualBasedEliminationBuilderAndSolverContinuity);

typedef BuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver> BaseType;

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



NodalResidualBasedEliminationBuilderAndSolverContinuity(
typename TLinearSolver::Pointer pNewLinearSystemSolver)
: BuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver>(pNewLinearSystemSolver)
{
}


~NodalResidualBasedEliminationBuilderAndSolverContinuity() override
{
}



void BuildNodally(
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
const ProcessInfo &CurrentProcessInfo = rModelPart.GetProcessInfo();

const unsigned int dimension = rModelPart.ElementsBegin()->GetGeometry().WorkingSpaceDimension();
const double timeInterval = CurrentProcessInfo[DELTA_TIME];
const double deviatoric_threshold = 0.1;
double pressure = 0;
double deltaPressure = 0;
double meanMeshSize = 0;
double characteristicLength = 0;
double density = 0;
double nodalVelocityNorm = 0;
double tauStab = 0;
double dNdXi = 0;
double dNdYi = 0;
double dNdZi = 0;
double dNdXj = 0;
double dNdYj = 0;
double dNdZj = 0;
unsigned int firstRow = 0;
unsigned int firstCol = 0;


{
ModelPart::NodeIterator NodesBegin;
ModelPart::NodeIterator NodesEnd;
OpenMPUtils::PartitionedIterators(rModelPart.Nodes(), NodesBegin, NodesEnd);

for (ModelPart::NodeIterator itNode = NodesBegin; itNode != NodesEnd; ++itNode)
{

NodeWeakPtrVectorType &neighb_nodes = itNode->GetValue(NEIGHBOUR_NODES);
const unsigned int neighSize = neighb_nodes.size() + 1;

if (neighSize > 1)
{

const double nodalVolume = itNode->FastGetSolutionStepValue(NODAL_VOLUME);
noalias(LHS_Contribution) = ZeroMatrix(neighSize, neighSize);
noalias(RHS_Contribution) = ZeroVector(neighSize);

if (EquationId.size() != neighSize)
EquationId.resize(neighSize, false);

double deviatoricCoeff = 0;
this->GetDeviatoricCoefficientForFluid(rModelPart, itNode, deviatoricCoeff);

if (deviatoricCoeff > deviatoric_threshold)
{
deviatoricCoeff = deviatoric_threshold;
}

double volumetricCoeff = timeInterval * itNode->FastGetSolutionStepValue(BULK_MODULUS);

deltaPressure = itNode->FastGetSolutionStepValue(PRESSURE, 0) - itNode->FastGetSolutionStepValue(PRESSURE, 1);

LHS_Contribution(0, 0) += nodalVolume / volumetricCoeff;
RHS_Contribution[0] += -deltaPressure * nodalVolume / volumetricCoeff;

RHS_Contribution[0] += itNode->GetSolutionStepValue(NODAL_VOLUMETRIC_DEF_RATE) * nodalVolume;

const unsigned int xDofPos = itNode->GetDofPosition(PRESSURE);
EquationId[0] = itNode->GetDof(PRESSURE, xDofPos).EquationId();

for (unsigned int i = 0; i < neighb_nodes.size(); i++)
{
EquationId[i + 1] = neighb_nodes[i].GetDof(PRESSURE, xDofPos).EquationId();
}

firstRow = 0;
firstCol = 0;
meanMeshSize = itNode->FastGetSolutionStepValue(NODAL_MEAN_MESH_SIZE);
characteristicLength = 1.0 * meanMeshSize;
density = itNode->FastGetSolutionStepValue(DENSITY);



if (dimension == 2)
{
nodalVelocityNorm = sqrt(itNode->FastGetSolutionStepValue(VELOCITY_X) * itNode->FastGetSolutionStepValue(VELOCITY_X) +
itNode->FastGetSolutionStepValue(VELOCITY_Y) * itNode->FastGetSolutionStepValue(VELOCITY_Y));
}
else if (dimension == 3)
{
nodalVelocityNorm = sqrt(itNode->FastGetSolutionStepValue(VELOCITY_X) * itNode->FastGetSolutionStepValue(VELOCITY_X) +
itNode->FastGetSolutionStepValue(VELOCITY_Y) * itNode->FastGetSolutionStepValue(VELOCITY_Y) +
itNode->FastGetSolutionStepValue(VELOCITY_Z) * itNode->FastGetSolutionStepValue(VELOCITY_Z));
}

tauStab = 1.0 * (characteristicLength * characteristicLength * timeInterval) / (density * nodalVelocityNorm * timeInterval * characteristicLength + density * characteristicLength * characteristicLength + 8.0 * deviatoricCoeff * timeInterval);
itNode->FastGetSolutionStepValue(NODAL_TAU) = tauStab;

LHS_Contribution(0, 0) += +nodalVolume * tauStab * density / (volumetricCoeff * timeInterval);
RHS_Contribution[0] += -nodalVolume * tauStab * density / (volumetricCoeff * timeInterval) * (deltaPressure - itNode->FastGetSolutionStepValue(PRESSURE_VELOCITY, 0) * timeInterval);

if (itNode->Is(FREE_SURFACE))
{
LHS_Contribution(0, 0) += +4.0 * tauStab * nodalVolume / (meanMeshSize * meanMeshSize);
RHS_Contribution[0] += -4.0 * tauStab * nodalVolume / (meanMeshSize * meanMeshSize) * itNode->FastGetSolutionStepValue(PRESSURE, 0);

const array_1d<double, 3> &Normal = itNode->FastGetSolutionStepValue(NORMAL);
Vector &SpatialDefRate = itNode->FastGetSolutionStepValue(NODAL_SPATIAL_DEF_RATE);
array_1d<double, 3> nodalAcceleration = 0.5 * (itNode->FastGetSolutionStepValue(VELOCITY, 0) - itNode->FastGetSolutionStepValue(VELOCITY, 1)) / timeInterval - itNode->FastGetSolutionStepValue(ACCELERATION, 1);


double nodalNormalAcceleration = 0;
double nodalNormalProjDefRate = 0;
if (dimension == 2)
{
nodalNormalProjDefRate = Normal[0] * SpatialDefRate[0] * Normal[0] + Normal[1] * SpatialDefRate[1] * Normal[1] + 2 * Normal[0] * SpatialDefRate[2] * Normal[1];

nodalNormalAcceleration = Normal[0] * nodalAcceleration[0] + Normal[1] * nodalAcceleration[1];
}
else if (dimension == 3)
{
nodalNormalProjDefRate = Normal[0] * SpatialDefRate[0] * Normal[0] + Normal[1] * SpatialDefRate[1] * Normal[1] + Normal[2] * SpatialDefRate[2] * Normal[2] +
2 * Normal[0] * SpatialDefRate[3] * Normal[1] + 2 * Normal[0] * SpatialDefRate[4] * Normal[2] + 2 * Normal[1] * SpatialDefRate[5] * Normal[2];



}
double accelerationContribution = 2.0 * density * nodalNormalAcceleration / meanMeshSize;
double deviatoricContribution = 8.0 * deviatoricCoeff * nodalNormalProjDefRate / (meanMeshSize * meanMeshSize);

RHS_Contribution[0] += 1.0 * tauStab * (accelerationContribution - deviatoricContribution) * nodalVolume;
}

array_1d<double, 3> &VolumeAcceleration = itNode->FastGetSolutionStepValue(VOLUME_ACCELERATION);












for (unsigned int i = 0; i < neighSize; i++)
{

dNdXi = itNode->FastGetSolutionStepValue(NODAL_SFD_NEIGHBOURS)[firstCol];
dNdYi = itNode->FastGetSolutionStepValue(NODAL_SFD_NEIGHBOURS)[firstCol + 1];

if (i != 0)
{
EquationId[i] = neighb_nodes[i - 1].GetDof(PRESSURE, xDofPos).EquationId();
density = neighb_nodes[i - 1].FastGetSolutionStepValue(DENSITY);











}

if (dimension == 2)
{
RHS_Contribution[i] += -tauStab * density * (dNdXi * VolumeAcceleration[0] + dNdYi * VolumeAcceleration[1]) * nodalVolume;
}
else if (dimension == 3)
{
dNdZi = itNode->FastGetSolutionStepValue(NODAL_SFD_NEIGHBOURS)[firstCol + 2];
RHS_Contribution[i] += -tauStab * density * (dNdXi * VolumeAcceleration[0] + dNdYi * VolumeAcceleration[1] + dNdZi * VolumeAcceleration[2]) * nodalVolume;
}

firstRow = 0;

for (unsigned int j = 0; j < neighSize; j++)
{
dNdXj = itNode->FastGetSolutionStepValue(NODAL_SFD_NEIGHBOURS)[firstRow];
dNdYj = itNode->FastGetSolutionStepValue(NODAL_SFD_NEIGHBOURS)[firstRow + 1];
if (j != 0)
{
pressure = neighb_nodes[j - 1].FastGetSolutionStepValue(PRESSURE, 0);
}
else
{
pressure = itNode->FastGetSolutionStepValue(PRESSURE, 0);
}


if (dimension == 2)
{
LHS_Contribution(i, j) += +tauStab * (dNdXi * dNdXj + dNdYi * dNdYj) * nodalVolume;
RHS_Contribution[i] += -tauStab * (dNdXi * dNdXj + dNdYi * dNdYj) * nodalVolume * pressure;

}
else if (dimension == 3)
{
dNdZj = itNode->FastGetSolutionStepValue(NODAL_SFD_NEIGHBOURS)[firstRow + 2];
LHS_Contribution(i, j) += +tauStab * (dNdXi * dNdXj + dNdYi * dNdYj + dNdZi * dNdZj) * nodalVolume;
RHS_Contribution[i] += -tauStab * (dNdXi * dNdXj + dNdYi * dNdYj + dNdZi * dNdZj) * nodalVolume * pressure;
}
firstRow += dimension;
}

firstCol += dimension;
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

void GetDeviatoricCoefficientForFluid(ModelPart &rModelPart, ModelPart::NodeIterator itNode, double &deviatoricCoefficient)
{
const double tolerance = 1e-12;

if (rModelPart.GetNodalSolutionStepVariablesList().Has(STATIC_FRICTION)) 
{
const double static_friction = itNode->FastGetSolutionStepValue(STATIC_FRICTION);
const double dynamic_friction = itNode->FastGetSolutionStepValue(DYNAMIC_FRICTION);
const double delta_friction = dynamic_friction - static_friction;
const double inertial_number_zero = itNode->FastGetSolutionStepValue(INERTIAL_NUMBER_ZERO);
const double grain_diameter = itNode->FastGetSolutionStepValue(GRAIN_DIAMETER);
const double grain_density = itNode->FastGetSolutionStepValue(GRAIN_DENSITY);
const double regularization_coeff = itNode->FastGetSolutionStepValue(REGULARIZATION_COEFFICIENT);

const double theta = 0.5;
double mean_pressure = itNode->FastGetSolutionStepValue(PRESSURE, 0) * theta + itNode->FastGetSolutionStepValue(PRESSURE, 1) * (1 - theta);

double pressure_tolerance = -1.0e-07;
if (mean_pressure > pressure_tolerance)
{
mean_pressure = pressure_tolerance;
}

const double equivalent_strain_rate = itNode->FastGetSolutionStepValue(NODAL_EQUIVALENT_STRAIN_RATE);
const double exponent = -equivalent_strain_rate / regularization_coeff;
const double second_viscous_term = delta_friction * grain_diameter / (inertial_number_zero * std::sqrt(std::fabs(mean_pressure) / grain_density) + equivalent_strain_rate * grain_diameter);

if (std::fabs(equivalent_strain_rate) > tolerance)
{
const double first_viscous_term = static_friction * (1 - std::exp(exponent)) / equivalent_strain_rate;
deviatoricCoefficient = (first_viscous_term + second_viscous_term) * std::fabs(mean_pressure);
}
else
{
deviatoricCoefficient = 1.0; 
}
}
else if (rModelPart.GetNodalSolutionStepVariablesList().Has(INTERNAL_FRICTION_ANGLE)) 
{
const double dynamic_viscosity = itNode->FastGetSolutionStepValue(DYNAMIC_VISCOSITY);
const double friction_angle = itNode->FastGetSolutionStepValue(INTERNAL_FRICTION_ANGLE);
const double cohesion = itNode->FastGetSolutionStepValue(COHESION);
const double adaptive_exponent = itNode->FastGetSolutionStepValue(ADAPTIVE_EXPONENT);

const double theta = 0.5;
double mean_pressure = itNode->FastGetSolutionStepValue(PRESSURE, 0) * theta + itNode->FastGetSolutionStepValue(PRESSURE, 1) * (1 - theta);

double pressure_tolerance = -1.0e-07;
if (mean_pressure > pressure_tolerance)
{
mean_pressure = pressure_tolerance;
}

const double equivalent_strain_rate = itNode->FastGetSolutionStepValue(NODAL_EQUIVALENT_STRAIN_RATE);

if (std::fabs(equivalent_strain_rate) > tolerance)
{
const double friction_angle_rad = friction_angle * Globals::Pi / 180.0;
const double tanFi = std::tan(friction_angle_rad);
double regularization = 1.0 - std::exp(-adaptive_exponent * equivalent_strain_rate);
deviatoricCoefficient = dynamic_viscosity + regularization * ((cohesion + tanFi * fabs(mean_pressure)) / equivalent_strain_rate);
}
else
{
deviatoricCoefficient = dynamic_viscosity;
}
}
else if (rModelPart.GetNodalSolutionStepVariablesList().Has(YIELD_SHEAR)) 
{
const double yieldShear = itNode->FastGetSolutionStepValue(YIELD_SHEAR);
const double equivalentStrainRate = itNode->FastGetSolutionStepValue(NODAL_EQUIVALENT_STRAIN_RATE);
const double adaptiveExponent = itNode->FastGetSolutionStepValue(ADAPTIVE_EXPONENT);
const double exponent = -adaptiveExponent * equivalentStrainRate;
deviatoricCoefficient = itNode->FastGetSolutionStepValue(DYNAMIC_VISCOSITY);
if (std::abs(equivalentStrainRate) > tolerance)
{
deviatoricCoefficient += (yieldShear / equivalentStrainRate) * (1 - exp(exponent));
}
}
else if (rModelPart.GetNodalSolutionStepVariablesList().Has(DYNAMIC_VISCOSITY))
{
deviatoricCoefficient = itNode->FastGetSolutionStepValue(DYNAMIC_VISCOSITY);
}
itNode->FastGetSolutionStepValue(DEVIATORIC_COEFFICIENT) = deviatoricCoefficient;
}

void BuildNodallyUnlessLaplacian(
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
const ProcessInfo &CurrentProcessInfo = rModelPart.GetProcessInfo();

const unsigned int dimension = rModelPart.ElementsBegin()->GetGeometry().WorkingSpaceDimension();
const double timeInterval = CurrentProcessInfo[DELTA_TIME];
const double deviatoric_threshold = 0.1;
double deltaPressure = 0;
double meanMeshSize = 0;
double characteristicLength = 0;
double density = 0;
double nodalVelocityNorm = 0;
double tauStab = 0;
double dNdXi = 0;
double dNdYi = 0;
double dNdZi = 0;
unsigned int firstCol = 0;


{
ModelPart::NodeIterator NodesBegin;
ModelPart::NodeIterator NodesEnd;
OpenMPUtils::PartitionedIterators(rModelPart.Nodes(), NodesBegin, NodesEnd);

for (ModelPart::NodeIterator itNode = NodesBegin; itNode != NodesEnd; ++itNode)
{

NodeWeakPtrVectorType &neighb_nodes = itNode->GetValue(NEIGHBOUR_NODES);
const unsigned int neighSize = neighb_nodes.size() + 1;

if (neighSize > 1)
{

const double nodalVolume = itNode->FastGetSolutionStepValue(NODAL_VOLUME);

noalias(LHS_Contribution) = ZeroMatrix(neighSize, neighSize);
noalias(RHS_Contribution) = ZeroVector(neighSize);

if (EquationId.size() != neighSize)
EquationId.resize(neighSize, false);

double deviatoricCoeff = 0;
this->GetDeviatoricCoefficientForFluid(rModelPart, itNode, deviatoricCoeff);

if (deviatoricCoeff > deviatoric_threshold)
{
deviatoricCoeff = deviatoric_threshold;
}

double volumetricCoeff = timeInterval * itNode->FastGetSolutionStepValue(BULK_MODULUS);

deltaPressure = itNode->FastGetSolutionStepValue(PRESSURE, 0) - itNode->FastGetSolutionStepValue(PRESSURE, 1);

LHS_Contribution(0, 0) += nodalVolume / volumetricCoeff;

RHS_Contribution[0] += -deltaPressure * nodalVolume / volumetricCoeff;

RHS_Contribution[0] += itNode->GetSolutionStepValue(NODAL_VOLUMETRIC_DEF_RATE) * nodalVolume;

const unsigned int xDofPos = itNode->GetDofPosition(PRESSURE);

EquationId[0] = itNode->GetDof(PRESSURE, xDofPos).EquationId();

for (unsigned int i = 0; i < neighb_nodes.size(); i++)
{
EquationId[i + 1] = neighb_nodes[i].GetDof(PRESSURE, xDofPos).EquationId();
}

firstCol = 0;
meanMeshSize = itNode->FastGetSolutionStepValue(NODAL_MEAN_MESH_SIZE);
characteristicLength = 1.0 * meanMeshSize;
density = itNode->FastGetSolutionStepValue(DENSITY);



if (dimension == 2)
{
nodalVelocityNorm = sqrt(itNode->FastGetSolutionStepValue(VELOCITY_X) * itNode->FastGetSolutionStepValue(VELOCITY_X) +
itNode->FastGetSolutionStepValue(VELOCITY_Y) * itNode->FastGetSolutionStepValue(VELOCITY_Y));
}
else if (dimension == 3)
{
nodalVelocityNorm = sqrt(itNode->FastGetSolutionStepValue(VELOCITY_X) * itNode->FastGetSolutionStepValue(VELOCITY_X) +
itNode->FastGetSolutionStepValue(VELOCITY_Y) * itNode->FastGetSolutionStepValue(VELOCITY_Y) +
itNode->FastGetSolutionStepValue(VELOCITY_Z) * itNode->FastGetSolutionStepValue(VELOCITY_Z));
}

tauStab = 1.0 * (characteristicLength * characteristicLength * timeInterval) /
(density * nodalVelocityNorm * timeInterval * characteristicLength + density * characteristicLength * characteristicLength + 8.0 * deviatoricCoeff * timeInterval);

itNode->FastGetSolutionStepValue(NODAL_TAU) = tauStab;

LHS_Contribution(0, 0) += +nodalVolume * tauStab * density / (volumetricCoeff * timeInterval);
RHS_Contribution[0] += -nodalVolume * tauStab * density / (volumetricCoeff * timeInterval) * (deltaPressure - itNode->FastGetSolutionStepValue(PRESSURE_VELOCITY, 0) * timeInterval);

if (itNode->Is(FREE_SURFACE))
{
LHS_Contribution(0, 0) += +4.0 * tauStab * nodalVolume / (meanMeshSize * meanMeshSize);
RHS_Contribution[0] += -4.0 * tauStab * nodalVolume / (meanMeshSize * meanMeshSize) * itNode->FastGetSolutionStepValue(PRESSURE, 0);

array_1d<double, 3> &Normal = itNode->FastGetSolutionStepValue(NORMAL);
Vector &SpatialDefRate = itNode->FastGetSolutionStepValue(NODAL_SPATIAL_DEF_RATE);
array_1d<double, 3> nodalAcceleration = 0.5 * (itNode->FastGetSolutionStepValue(VELOCITY, 0) - itNode->FastGetSolutionStepValue(VELOCITY, 1)) / timeInterval - itNode->FastGetSolutionStepValue(ACCELERATION, 1);


double nodalNormalAcceleration = 0;
double nodalNormalProjDefRate = 0;
if (dimension == 2)
{
nodalNormalProjDefRate = Normal[0] * SpatialDefRate[0] * Normal[0] + Normal[1] * SpatialDefRate[1] * Normal[1] + 2 * Normal[0] * SpatialDefRate[2] * Normal[1];

nodalNormalAcceleration = Normal[0] * nodalAcceleration[0] + Normal[1] * nodalAcceleration[1];
}
else if (dimension == 3)
{
nodalNormalProjDefRate = Normal[0] * SpatialDefRate[0] * Normal[0] + Normal[1] * SpatialDefRate[1] * Normal[1] + Normal[2] * SpatialDefRate[2] * Normal[2] +
2 * Normal[0] * SpatialDefRate[3] * Normal[1] + 2 * Normal[0] * SpatialDefRate[4] * Normal[2] + 2 * Normal[1] * SpatialDefRate[5] * Normal[2];



}
double accelerationContribution = 2.0 * density * nodalNormalAcceleration / meanMeshSize;
double deviatoricContribution = 8.0 * deviatoricCoeff * nodalNormalProjDefRate / (meanMeshSize * meanMeshSize);

RHS_Contribution[0] += 1.0 * tauStab * (accelerationContribution - deviatoricContribution) * nodalVolume;
}

array_1d<double, 3> &VolumeAcceleration = itNode->FastGetSolutionStepValue(VOLUME_ACCELERATION);












for (unsigned int i = 0; i < neighSize; i++)
{

dNdXi = itNode->FastGetSolutionStepValue(NODAL_SFD_NEIGHBOURS)[firstCol];
dNdYi = itNode->FastGetSolutionStepValue(NODAL_SFD_NEIGHBOURS)[firstCol + 1];

if (i != 0)
{
EquationId[i] = neighb_nodes[i - 1].GetDof(PRESSURE, xDofPos).EquationId();
density = neighb_nodes[i - 1].FastGetSolutionStepValue(DENSITY);












}

if (dimension == 2)
{
RHS_Contribution[i] += -tauStab * density * (dNdXi * VolumeAcceleration[0] + dNdYi * VolumeAcceleration[1]) * nodalVolume;
}
else if (dimension == 3)
{
dNdZi = itNode->FastGetSolutionStepValue(NODAL_SFD_NEIGHBOURS)[firstCol + 2];
RHS_Contribution[i] += -tauStab * density * (dNdXi * VolumeAcceleration[0] + dNdYi * VolumeAcceleration[1] + dNdZi * VolumeAcceleration[2]) * nodalVolume;
}

firstCol += dimension;
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

void BuildNodallyNoVolumetricStabilizedTerms(
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
const ProcessInfo &CurrentProcessInfo = rModelPart.GetProcessInfo();

const unsigned int dimension = rModelPart.ElementsBegin()->GetGeometry().WorkingSpaceDimension();
const double timeInterval = CurrentProcessInfo[DELTA_TIME];
const double deviatoric_threshold = 0.1;
double deltaPressure = 0;
double meanMeshSize = 0;
double characteristicLength = 0;
double density = 0;
double nodalVelocityNorm = 0;
double tauStab = 0;


{
ModelPart::NodeIterator NodesBegin;
ModelPart::NodeIterator NodesEnd;
OpenMPUtils::PartitionedIterators(rModelPart.Nodes(), NodesBegin, NodesEnd);

for (ModelPart::NodeIterator itNode = NodesBegin; itNode != NodesEnd; ++itNode)
{

NodeWeakPtrVectorType &neighb_nodes = itNode->GetValue(NEIGHBOUR_NODES);
const unsigned int neighSize = neighb_nodes.size() + 1;

if (neighSize > 1)
{

const double nodalVolume = itNode->FastGetSolutionStepValue(NODAL_VOLUME);

noalias(LHS_Contribution) = ZeroMatrix(neighSize, neighSize);
noalias(RHS_Contribution) = ZeroVector(neighSize);

if (EquationId.size() != neighSize)
EquationId.resize(neighSize, false);

double deviatoricCoeff = 0;
this->GetDeviatoricCoefficientForFluid(rModelPart, itNode, deviatoricCoeff);

if (deviatoricCoeff > deviatoric_threshold)
{
deviatoricCoeff = deviatoric_threshold;
}

double volumetricCoeff = timeInterval * itNode->FastGetSolutionStepValue(BULK_MODULUS);

deltaPressure = itNode->FastGetSolutionStepValue(PRESSURE, 0) - itNode->FastGetSolutionStepValue(PRESSURE, 1);

LHS_Contribution(0, 0) += nodalVolume / volumetricCoeff;

RHS_Contribution[0] += -deltaPressure * nodalVolume / volumetricCoeff;

RHS_Contribution[0] += itNode->GetSolutionStepValue(NODAL_VOLUMETRIC_DEF_RATE) * nodalVolume;

const unsigned int xDofPos = itNode->GetDofPosition(PRESSURE);

EquationId[0] = itNode->GetDof(PRESSURE, xDofPos).EquationId();

for (unsigned int i = 0; i < neighb_nodes.size(); i++)
{
EquationId[i + 1] = neighb_nodes[i].GetDof(PRESSURE, xDofPos).EquationId();
}

meanMeshSize = itNode->FastGetSolutionStepValue(NODAL_MEAN_MESH_SIZE);
characteristicLength = 1.0 * meanMeshSize;
density = itNode->FastGetSolutionStepValue(DENSITY);



if (dimension == 2)
{
nodalVelocityNorm = sqrt(itNode->FastGetSolutionStepValue(VELOCITY_X) * itNode->FastGetSolutionStepValue(VELOCITY_X) +
itNode->FastGetSolutionStepValue(VELOCITY_Y) * itNode->FastGetSolutionStepValue(VELOCITY_Y));
}
else if (dimension == 3)
{
nodalVelocityNorm = sqrt(itNode->FastGetSolutionStepValue(VELOCITY_X) * itNode->FastGetSolutionStepValue(VELOCITY_X) +
itNode->FastGetSolutionStepValue(VELOCITY_Y) * itNode->FastGetSolutionStepValue(VELOCITY_Y) +
itNode->FastGetSolutionStepValue(VELOCITY_Z) * itNode->FastGetSolutionStepValue(VELOCITY_Z));
}

tauStab = 1.0 * (characteristicLength * characteristicLength * timeInterval) /
(density * nodalVelocityNorm * timeInterval * characteristicLength + density * characteristicLength * characteristicLength + 8.0 * deviatoricCoeff * timeInterval);

itNode->FastGetSolutionStepValue(NODAL_TAU) = tauStab;

LHS_Contribution(0, 0) += +nodalVolume * tauStab * density / (volumetricCoeff * timeInterval);
RHS_Contribution[0] += -nodalVolume * tauStab * density / (volumetricCoeff * timeInterval) * (deltaPressure - itNode->FastGetSolutionStepValue(PRESSURE_VELOCITY, 0) * timeInterval);

if (itNode->Is(FREE_SURFACE))
{

LHS_Contribution(0, 0) += +4.0 * tauStab * nodalVolume / (meanMeshSize * meanMeshSize);
RHS_Contribution[0] += -4.0 * tauStab * nodalVolume / (meanMeshSize * meanMeshSize) * itNode->FastGetSolutionStepValue(PRESSURE, 0);

array_1d<double, 3> &Normal = itNode->FastGetSolutionStepValue(NORMAL);
Vector &SpatialDefRate = itNode->FastGetSolutionStepValue(NODAL_SPATIAL_DEF_RATE);
array_1d<double, 3> nodalAcceleration = 0.5 * (itNode->FastGetSolutionStepValue(VELOCITY, 0) - itNode->FastGetSolutionStepValue(VELOCITY, 1)) / timeInterval - itNode->FastGetSolutionStepValue(ACCELERATION, 1);


double nodalNormalAcceleration = 0;
double nodalNormalProjDefRate = 0;
if (dimension == 2)
{
nodalNormalProjDefRate = Normal[0] * SpatialDefRate[0] * Normal[0] + Normal[1] * SpatialDefRate[1] * Normal[1] + 2 * Normal[0] * SpatialDefRate[2] * Normal[1];

nodalNormalAcceleration = Normal[0] * nodalAcceleration[0] + Normal[1] * nodalAcceleration[1];
}
else if (dimension == 3)
{
nodalNormalProjDefRate = Normal[0] * SpatialDefRate[0] * Normal[0] + Normal[1] * SpatialDefRate[1] * Normal[1] + Normal[2] * SpatialDefRate[2] * Normal[2] +
2 * Normal[0] * SpatialDefRate[3] * Normal[1] + 2 * Normal[0] * SpatialDefRate[4] * Normal[2] + 2 * Normal[1] * SpatialDefRate[5] * Normal[2];



}
double accelerationContribution = 2.0 * density * nodalNormalAcceleration / meanMeshSize;
double deviatoricContribution = 8.0 * deviatoricCoeff * nodalNormalProjDefRate / (meanMeshSize * meanMeshSize);

RHS_Contribution[0] += 1.0 * tauStab * (accelerationContribution - deviatoricContribution) * nodalVolume;
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

void BuildNodallyNotStabilized(
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
const ProcessInfo &CurrentProcessInfo = rModelPart.GetProcessInfo();

const double timeInterval = CurrentProcessInfo[DELTA_TIME];
const double deviatoric_threshold = 0.1;
double deltaPressure = 0;


{
ModelPart::NodeIterator NodesBegin;
ModelPart::NodeIterator NodesEnd;
OpenMPUtils::PartitionedIterators(rModelPart.Nodes(), NodesBegin, NodesEnd);

for (ModelPart::NodeIterator itNode = NodesBegin; itNode != NodesEnd; ++itNode)
{

NodeWeakPtrVectorType &neighb_nodes = itNode->GetValue(NEIGHBOUR_NODES);
const unsigned int neighSize = neighb_nodes.size() + 1;

if (neighSize > 1)
{

const double nodalVolume = itNode->FastGetSolutionStepValue(NODAL_VOLUME);

noalias(LHS_Contribution) = ZeroMatrix(neighSize, neighSize);
noalias(RHS_Contribution) = ZeroVector(neighSize);

if (EquationId.size() != neighSize)
EquationId.resize(neighSize, false);

double deviatoricCoeff = 0;
this->GetDeviatoricCoefficientForFluid(rModelPart, itNode, deviatoricCoeff);

if (deviatoricCoeff > deviatoric_threshold)
{
deviatoricCoeff = deviatoric_threshold;
}

double volumetricCoeff = timeInterval * itNode->FastGetSolutionStepValue(BULK_MODULUS);

deltaPressure = itNode->FastGetSolutionStepValue(PRESSURE, 0) - itNode->FastGetSolutionStepValue(PRESSURE, 1);

LHS_Contribution(0, 0) += nodalVolume / volumetricCoeff;

RHS_Contribution[0] += -deltaPressure * nodalVolume / volumetricCoeff;

RHS_Contribution[0] += itNode->GetSolutionStepValue(NODAL_VOLUMETRIC_DEF_RATE) * nodalVolume;

const unsigned int xDofPos = itNode->GetDofPosition(PRESSURE);

EquationId[0] = itNode->GetDof(PRESSURE, xDofPos).EquationId();

for (unsigned int i = 0; i < neighb_nodes.size(); i++)
{
EquationId[i + 1] = neighb_nodes[i].GetDof(PRESSURE, xDofPos).EquationId();
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

void BuildAll(
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

const ProcessInfo &CurrentProcessInfo = rModelPart.GetProcessInfo();

const double timeInterval = CurrentProcessInfo[DELTA_TIME];
const double deviatoric_threshold = 0.1;
double deltaPressure = 0;


ModelPart::NodeIterator NodesBegin;
ModelPart::NodeIterator NodesEnd;
OpenMPUtils::PartitionedIterators(rModelPart.Nodes(), NodesBegin, NodesEnd);

for (ModelPart::NodeIterator itNode = NodesBegin; itNode != NodesEnd; ++itNode)
{

NodeWeakPtrVectorType &neighb_nodes = itNode->GetValue(NEIGHBOUR_NODES);
const unsigned int neighSize = neighb_nodes.size() + 1;

if (neighSize > 1)
{

if (LHS_Contribution.size1() != 1)
LHS_Contribution.resize(1, 1, false); 

if (RHS_Contribution.size() != 1)
RHS_Contribution.resize(1, false); 

noalias(LHS_Contribution) = ZeroMatrix(1, 1);
noalias(RHS_Contribution) = ZeroVector(1);

if (EquationId.size() != 1)
EquationId.resize(1, false);

double nodalVolume = itNode->FastGetSolutionStepValue(NODAL_VOLUME);

if (nodalVolume > 0)
{ 

double deviatoricCoeff = 0;
this->GetDeviatoricCoefficientForFluid(rModelPart, itNode, deviatoricCoeff);

if (deviatoricCoeff > deviatoric_threshold)
{
deviatoricCoeff = deviatoric_threshold;
}

double volumetricCoeff = timeInterval * itNode->FastGetSolutionStepValue(BULK_MODULUS);

deltaPressure = itNode->FastGetSolutionStepValue(PRESSURE, 0) - itNode->FastGetSolutionStepValue(PRESSURE, 1);

LHS_Contribution(0, 0) += nodalVolume / volumetricCoeff;

RHS_Contribution[0] += -deltaPressure * nodalVolume / volumetricCoeff;

RHS_Contribution[0] += itNode->GetSolutionStepValue(NODAL_VOLUMETRIC_DEF_RATE) * nodalVolume;
}

const unsigned int xDofPos = itNode->GetDofPosition(PRESSURE);

EquationId[0] = itNode->GetDof(PRESSURE, xDofPos).EquationId();

#ifdef _OPENMP
Assemble(A, b, LHS_Contribution, RHS_Contribution, EquationId, mlock_array);
#else
Assemble(A, b, LHS_Contribution, RHS_Contribution, EquationId);
#endif
}
}


ElementsArrayType &pElements = rModelPart.Elements();
int number_of_threads = ParallelUtilities::GetNumThreads();

#ifdef _OPENMP
int A_size = A.size1();

std::vector<omp_lock_t> lock_array(A.size1());

for (int i = 0; i < A_size; i++)
omp_init_lock(&lock_array[i]);
#endif

DenseVector<unsigned int> element_partition;
CreatePartition(number_of_threads, pElements.size(), element_partition);
if (this->GetEchoLevel() > 0)
{
KRATOS_WATCH(number_of_threads);
KRATOS_WATCH(element_partition);
}

#pragma omp parallel for firstprivate(number_of_threads) schedule(static, 1)
for (int k = 0; k < number_of_threads; k++)
{
LocalSystemMatrixType elementalLHS_Contribution = LocalSystemMatrixType(0, 0);
LocalSystemVectorType elementalRHS_Contribution = LocalSystemVectorType(0);

Element::EquationIdVectorType elementalEquationId;
const ProcessInfo &CurrentProcessInfo = rModelPart.GetProcessInfo();
typename ElementsArrayType::ptr_iterator it_begin = pElements.ptr_begin() + element_partition[k];
typename ElementsArrayType::ptr_iterator it_end = pElements.ptr_begin() + element_partition[k + 1];

unsigned int pos = (rModelPart.Nodes().begin())->GetDofPosition(PRESSURE);

for (typename ElementsArrayType::ptr_iterator it = it_begin; it != it_end; ++it)
{
(*it)->CalculateLocalSystem(elementalLHS_Contribution, elementalRHS_Contribution, CurrentProcessInfo);

Geometry<Node> &geom = (*it)->GetGeometry();
if (elementalEquationId.size() != geom.size())
elementalEquationId.resize(geom.size(), false);

for (unsigned int i = 0; i < geom.size(); i++)
elementalEquationId[i] = geom[i].GetDof(PRESSURE, pos).EquationId();

#ifdef _OPENMP
this->Assemble(A, b, elementalLHS_Contribution, elementalRHS_Contribution, elementalEquationId, lock_array);
#else
this->Assemble(A, b, elementalLHS_Contribution, elementalRHS_Contribution, elementalEquationId);
#endif
}
}

#ifdef _OPENMP
for (int i = 0; i < A_size; i++)
omp_destroy_lock(&lock_array[i]);
#endif

KRATOS_CATCH("")
}

void SystemSolve(
TSystemMatrixType &A,
TSystemVectorType &Dx,
TSystemVectorType &b) override
{
KRATOS_TRY

double norm_b;
if (TSparseSpace::Size(b) != 0)
norm_b = TSparseSpace::TwoNorm(b);
else
norm_b = 0.00;

if (norm_b != 0.00)
{
BaseType::mpLinearSystemSolver->Solve(A, Dx, b);
}
else
TSparseSpace::SetToZero(Dx);

KRATOS_INFO_IF("NodalResidualBasedEliminationBuilderAndSolverContinuity", this->GetEchoLevel() > 1) << *(BaseType::mpLinearSystemSolver) << std::endl;

KRATOS_CATCH("")
}


void SystemSolveWithPhysics(
TSystemMatrixType &A,
TSystemVectorType &Dx,
TSystemVectorType &b,
ModelPart &rModelPart)
{
KRATOS_TRY

double norm_b;
if (TSparseSpace::Size(b) != 0)
norm_b = TSparseSpace::TwoNorm(b);
else
norm_b = 0.00;

if (norm_b != 0.00)
{
if (BaseType::mpLinearSystemSolver->AdditionalPhysicalDataIsNeeded())
BaseType::mpLinearSystemSolver->ProvideAdditionalData(A, Dx, b, BaseType::mDofSet, rModelPart);

BaseType::mpLinearSystemSolver->Solve(A, Dx, b);
}
else
{
TSparseSpace::SetToZero(Dx);
KRATOS_WARNING_IF("NodalResidualBasedEliminationBuilderAndSolverContinuity", rModelPart.GetCommunicator().MyPID() == 0) << "ATTENTION! setting the RHS to zero!" << std::endl;
}

KRATOS_INFO_IF("NodalResidualBasedEliminationBuilderAndSolverContinuity", this->GetEchoLevel() > 1 && rModelPart.GetCommunicator().MyPID() == 0) << *(BaseType::mpLinearSystemSolver) << std::endl;

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







BuildAll(pScheme, rModelPart, A, b);


Timer::Stop("Build");


ApplyDirichletConditions(pScheme, rModelPart, A, Dx, b);

KRATOS_INFO_IF("ResidualBasedBlockBuilderAndSolver", (this->GetEchoLevel() == 3)) << "Before the solution of the system"
<< "\nSystem Matrix = " << A << "\nUnknowns vector = " << Dx << "\nRHS vector = " << b << std::endl;


Timer::Start("Solve");


SystemSolveWithPhysics(A, Dx, b, rModelPart);


Timer::Stop("Solve");


KRATOS_INFO_IF("ResidualBasedBlockBuilderAndSolver", (this->GetEchoLevel() == 3)) << "After the solution of the system"
<< "\nSystem Matrix = " << A << "\nUnknowns vector = " << Dx << "\nRHS vector = " << b << std::endl;

KRATOS_CATCH("")
}

void Build(
typename TSchemeType::Pointer pScheme,
ModelPart &r_model_part,
TSystemMatrixType &A,
TSystemVectorType &b) override
{
KRATOS_TRY
if (!pScheme)
KRATOS_THROW_ERROR(std::runtime_error, "No scheme provided!", "");

ElementsArrayType &pElements = r_model_part.Elements();


TSparseSpace::SetToZero(*(BaseType::mpReactionsVector));

int number_of_threads = ParallelUtilities::GetNumThreads();

#ifdef _OPENMP
int A_size = A.size1();

std::vector<omp_lock_t> lock_array(A.size1());

for (int i = 0; i < A_size; i++)
omp_init_lock(&lock_array[i]);
#endif

DenseVector<unsigned int> element_partition;
CreatePartition(number_of_threads, pElements.size(), element_partition);
if (this->GetEchoLevel() > 0)
{
KRATOS_WATCH(number_of_threads);
KRATOS_WATCH(element_partition);
}


#pragma omp parallel for firstprivate(number_of_threads) schedule(static, 1)
for (int k = 0; k < number_of_threads; k++)
{
LocalSystemMatrixType LHS_Contribution = LocalSystemMatrixType(0, 0);
LocalSystemVectorType RHS_Contribution = LocalSystemVectorType(0);

Element::EquationIdVectorType EquationId;
const ProcessInfo &CurrentProcessInfo = r_model_part.GetProcessInfo();
typename ElementsArrayType::ptr_iterator it_begin = pElements.ptr_begin() + element_partition[k];
typename ElementsArrayType::ptr_iterator it_end = pElements.ptr_begin() + element_partition[k + 1];

unsigned int pos = (r_model_part.Nodes().begin())->GetDofPosition(PRESSURE);

for (typename ElementsArrayType::ptr_iterator it = it_begin; it != it_end; ++it)
{

(*it)->CalculateLocalSystem(LHS_Contribution, RHS_Contribution, CurrentProcessInfo);

Geometry<Node> &geom = (*it)->GetGeometry();
if (EquationId.size() != geom.size())
EquationId.resize(geom.size(), false);

for (unsigned int i = 0; i < geom.size(); i++)
EquationId[i] = geom[i].GetDof(PRESSURE, pos).EquationId();

#ifdef _OPENMP
this->Assemble(A, b, LHS_Contribution, RHS_Contribution, EquationId, lock_array);
#else
this->Assemble(A, b, LHS_Contribution, RHS_Contribution, EquationId);
#endif
}
}


#ifdef _OPENMP
for (int i = 0; i < A_size; i++)
omp_destroy_lock(&lock_array[i]);
#endif

KRATOS_CATCH("")
}


void SetUpDofSet(
typename TSchemeType::Pointer pScheme,
ModelPart &rModelPart) override
{
KRATOS_TRY;

KRATOS_INFO_IF("NodalResidualBasedEliminationBuilderAndSolverContinuity", this->GetEchoLevel() > 1 && rModelPart.GetCommunicator().MyPID() == 0) << "Setting up the dofs" << std::endl;

ElementsArrayType &pElements = rModelPart.Elements();
const int nelements = static_cast<int>(pElements.size());

Element::DofsVectorType ElementalDofList;

const ProcessInfo &CurrentProcessInfo = rModelPart.GetProcessInfo();

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
Doftemp.push_back((*it));
}
Doftemp.Sort();

BaseType::mDofSet = Doftemp;

KRATOS_ERROR_IF(BaseType::mDofSet.size() == 0) << "No degrees of freedom!" << std::endl;

BaseType::mDofSetIsInitialized = true;

KRATOS_INFO_IF("NodalResidualBasedEliminationBuilderAndSolverContinuity", this->GetEchoLevel() > 2 && rModelPart.GetCommunicator().MyPID() == 0) << "Finished setting up the dofs" << std::endl;

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


void SetUpSystem(
ModelPart &rModelPart) override
{
int free_id = 0;
int fix_id = BaseType::mDofSet.size();

for (typename DofsArrayType::iterator dof_iterator = BaseType::mDofSet.begin(); dof_iterator != BaseType::mDofSet.end(); ++dof_iterator)
if (dof_iterator->IsFixed())
dof_iterator->SetEquationId(--fix_id);
else
dof_iterator->SetEquationId(free_id++);

BaseType::mEquationSystemSize = fix_id;
}

/

if (pA == NULL) 
{
TSystemMatrixPointerType pNewA = TSystemMatrixPointerType(new TSystemMatrixType(0, 0));
pA.swap(pNewA);
}
if (pDx == NULL) 
{
TSystemVectorPointerType pNewDx = TSystemVectorPointerType(new TSystemVectorType(0));
pDx.swap(pNewDx);
}
if (pb == NULL) 
{
TSystemVectorPointerType pNewb = TSystemVectorPointerType(new TSystemVectorType(0));
pb.swap(pNewb);
}
if (BaseType::mpReactionsVector == NULL) 
{
TSystemVectorPointerType pNewReactionsVector = TSystemVectorPointerType(new TSystemVectorType(0));
BaseType::mpReactionsVector.swap(pNewReactionsVector);
}

TSystemMatrixType &A = *pA;
TSystemVectorType &Dx = *pDx;
TSystemVectorType &b = *pb;

if (A.size1() == 0 || BaseType::GetReshapeMatrixFlag() == true) 
{
A.resize(BaseType::mEquationSystemSize, BaseType::mEquationSystemSize, false);
ConstructMatrixStructure(pScheme, A, rModelPart);
}
else
{
if (A.size1() != BaseType::mEquationSystemSize || A.size2() != BaseType::mEquationSystemSize)
{
KRATOS_WATCH("it should not come here!!!!!!!! ... this is SLOW");
KRATOS_ERROR << "The equation system size has changed during the simulation. This is not permited." << std::endl;
A.resize(BaseType::mEquationSystemSize, BaseType::mEquationSystemSize, true);
ConstructMatrixStructure(pScheme, A, rModelPart);
}
}
if (Dx.size() != BaseType::mEquationSystemSize)
Dx.resize(BaseType::mEquationSystemSize, false);
if (b.size() != BaseType::mEquationSystemSize)
b.resize(BaseType::mEquationSystemSize, false);

if (BaseType::mCalculateReactionsFlag == true)
{
unsigned int ReactionsVectorSize = BaseType::mDofSet.size();
if (BaseType::mpReactionsVector->size() != ReactionsVectorSize)
BaseType::mpReactionsVector->resize(ReactionsVectorSize, false);
}


KRATOS_CATCH("")
}

/
void ApplyDirichletConditions(
typename TSchemeType::Pointer pScheme,
ModelPart &rModelPart,
TSystemMatrixType &A,
TSystemVectorType &Dx,
TSystemVectorType &b) override
{
}


void Clear() override
{
this->mDofSet = DofsArrayType();

if (this->mpReactionsVector != NULL)
TSparseSpace::Clear((this->mpReactionsVector));

this->mpLinearSystemSolver->Clear();

KRATOS_INFO_IF("NodalResidualBasedEliminationBuilderAndSolverContinuity", this->GetEchoLevel() > 1) << "Clear Function called" << std::endl;
}


int Check(ModelPart &rModelPart) override
{
KRATOS_TRY

return 0;
KRATOS_CATCH("");
}





protected:




void Assemble(
TSystemMatrixType &A,
TSystemVectorType &b,
const LocalSystemMatrixType &LHS_Contribution,
const LocalSystemVectorType &RHS_Contribution,
const Element::EquationIdVectorType &EquationId
#ifdef _OPENMP
,
std::vector<omp_lock_t> &lock_array
#endif
)
{
unsigned int local_size = LHS_Contribution.size1();

for (unsigned int i_local = 0; i_local < local_size; i_local++)
{
unsigned int i_global = EquationId[i_local];

if (i_global < BaseType::mEquationSystemSize)
{
#ifdef _OPENMP
omp_set_lock(&lock_array[i_global]);
#endif
b[i_global] += RHS_Contribution(i_local);
for (unsigned int j_local = 0; j_local < local_size; j_local++)
{
unsigned int j_global = EquationId[j_local];
if (j_global < BaseType::mEquationSystemSize)
{
A(i_global, j_global) += LHS_Contribution(i_local, j_local);
}
}
#ifdef _OPENMP
omp_unset_lock(&lock_array[i_global]);
#endif
}
}
}

/




} 

#endif 
