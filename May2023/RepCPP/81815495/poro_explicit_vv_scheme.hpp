
#if !defined(KRATOS_PORO_EXPLICIT_VV_SCHEME_HPP_INCLUDED)
#define KRATOS_PORO_EXPLICIT_VV_SCHEME_HPP_INCLUDED




#include "custom_strategies/schemes/poro_explicit_cd_scheme.hpp"
#include "utilities/variable_utils.h"

#include "poromechanics_application_variables.h"

namespace Kratos {








template <class TSparseSpace,
class TDenseSpace 
>
class PoroExplicitVVScheme
: public PoroExplicitCDScheme<TSparseSpace, TDenseSpace> {

public:

typedef Scheme<TSparseSpace, TDenseSpace> BaseofBaseType;
typedef PoroExplicitCDScheme<TSparseSpace, TDenseSpace> BaseType;

typedef typename BaseType::DofsArrayType DofsArrayType;
typedef typename BaseType::TSystemMatrixType TSystemMatrixType;
typedef typename BaseType::TSystemVectorType TSystemVectorType;
typedef typename BaseType::LocalSystemVectorType LocalSystemVectorType;

typedef ModelPart::ElementsContainerType ElementsArrayType;
typedef ModelPart::ConditionsContainerType ConditionsArrayType;
typedef ModelPart::NodesContainerType NodesArrayType;

typedef std::size_t SizeType;

typedef std::size_t IndexType;

typedef typename ModelPart::NodeIterator NodeIterator;

static constexpr double numerical_limit = std::numeric_limits<double>::epsilon();

using BaseType::mDeltaTime;
using BaseType::mAlpha;
using BaseType::mBeta;
using BaseType::mTheta;
using BaseType::mGCoefficient;

KRATOS_CLASS_POINTER_DEFINITION(PoroExplicitVVScheme);



PoroExplicitVVScheme()
: PoroExplicitCDScheme<TSparseSpace, TDenseSpace>()
{

}


virtual ~PoroExplicitVVScheme() {}



void InitializeExplicitScheme(
ModelPart& rModelPart,
const SizeType DomainSize = 3
) override
{
KRATOS_TRY

NodesArrayType& r_nodes = rModelPart.Nodes();

const auto it_node_begin = rModelPart.NodesBegin();

#pragma omp parallel for schedule(guided,512)
for (int i = 0; i < static_cast<int>(r_nodes.size()); ++i) {
auto it_node = (it_node_begin + i);
it_node->SetValue(NODAL_MASS, 0.0);
array_1d<double, 3>& r_force_residual = it_node->FastGetSolutionStepValue(FORCE_RESIDUAL);
double& r_flux_residual = it_node->FastGetSolutionStepValue(FLUX_RESIDUAL);
array_1d<double, 3>& r_external_force = it_node->FastGetSolutionStepValue(EXTERNAL_FORCE);
array_1d<double, 3>& r_internal_force = it_node->FastGetSolutionStepValue(INTERNAL_FORCE);
array_1d<double, 3>& r_damping_force = it_node->FastGetSolutionStepValue(DAMPING_FORCE);
noalias(r_force_residual) = ZeroVector(3);
r_flux_residual = 0.0;
noalias(r_external_force) = ZeroVector(3);
noalias(r_internal_force) = ZeroVector(3);
noalias(r_damping_force) = ZeroVector(3);
Matrix& rInitialStress = it_node->FastGetSolutionStepValue(INITIAL_STRESS_TENSOR);
if(rInitialStress.size1() != 3)
rInitialStress.resize(3,3,false);
noalias(rInitialStress) = ZeroMatrix(3,3);
}

KRATOS_CATCH("")
}


void InitializeResidual(ModelPart& rModelPart) override
{
KRATOS_TRY

NodesArrayType& r_nodes = rModelPart.Nodes();

const array_1d<double, 3> zero_array = ZeroVector(3);
VariableUtils().SetVariable(FORCE_RESIDUAL, zero_array,r_nodes);
VariableUtils().SetVariable(FLUX_RESIDUAL, 0.0,r_nodes);
VariableUtils().SetVariable(EXTERNAL_FORCE, zero_array,r_nodes);
VariableUtils().SetVariable(INTERNAL_FORCE, zero_array,r_nodes);
VariableUtils().SetVariable(DAMPING_FORCE, zero_array,r_nodes);

KRATOS_CATCH("")
}

void Predict(
ModelPart& rModelPart,
DofsArrayType& rDofSet,
TSystemMatrixType& A,
TSystemVectorType& Dx,
TSystemVectorType& b
) override
{
KRATOS_TRY;

this->CalculateAndAddRHS(rModelPart);

const ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();

NodesArrayType& r_nodes = rModelPart.Nodes();

const SizeType dim = r_current_process_info[DOMAIN_SIZE];

mDeltaTime = r_current_process_info[DELTA_TIME];

const auto it_node_begin = rModelPart.NodesBegin();

const IndexType disppos = it_node_begin->GetDofPosition(DISPLACEMENT_X);

#pragma omp parallel for schedule(guided,512)
for (int i = 0; i < static_cast<int>(r_nodes.size()); ++i) {
this->PredictTranslationalDegreesOfFreedom(it_node_begin + i, disppos, dim);
} 

InitializeResidual(rModelPart);

this->CalculateAndAddRHS(rModelPart);

KRATOS_CATCH("")
}


virtual void PredictTranslationalDegreesOfFreedom(
NodeIterator itCurrentNode,
const IndexType DisplacementPosition,
const SizeType DomainSize = 3
)
{
array_1d<double, 3>& r_displacement = itCurrentNode->FastGetSolutionStepValue(DISPLACEMENT);
array_1d<double, 3>& r_velocity = itCurrentNode->FastGetSolutionStepValue(VELOCITY);
const double nodal_mass = itCurrentNode->GetValue(NODAL_MASS);

const array_1d<double, 3>& r_external_force = itCurrentNode->FastGetSolutionStepValue(EXTERNAL_FORCE);
const array_1d<double, 3>& r_internal_force = itCurrentNode->FastGetSolutionStepValue(INTERNAL_FORCE);
const array_1d<double, 3>& r_damping_force = itCurrentNode->FastGetSolutionStepValue(DAMPING_FORCE);

std::array<bool, 3> fix_displacements = {false, false, false};
fix_displacements[0] = (itCurrentNode->GetDof(DISPLACEMENT_X, DisplacementPosition).IsFixed());
fix_displacements[1] = (itCurrentNode->GetDof(DISPLACEMENT_Y, DisplacementPosition + 1).IsFixed());
if (DomainSize == 3)
fix_displacements[2] = (itCurrentNode->GetDof(DISPLACEMENT_Z, DisplacementPosition + 2).IsFixed());

if (nodal_mass > numerical_limit){
for (IndexType j = 0; j < DomainSize; j++) {
if (fix_displacements[j] == false) {
r_displacement[j] += r_velocity[j]*mDeltaTime + 0.5 * (r_external_force[j]
- r_internal_force[j]
- r_damping_force[j])/nodal_mass * mDeltaTime * mDeltaTime;
r_velocity[j] += 0.5 * mDeltaTime * (r_external_force[j]
- r_internal_force[j]
- r_damping_force[j])/nodal_mass;
}
}
}
else {
noalias(r_displacement) = ZeroVector(3);
noalias(r_velocity) = ZeroVector(3);
}
}


void UpdateTranslationalDegreesOfFreedom(
NodeIterator itCurrentNode,
const IndexType DisplacementPosition,
const SizeType DomainSize = 3
) override
{
array_1d<double, 3>& r_velocity = itCurrentNode->FastGetSolutionStepValue(VELOCITY);
const double nodal_mass = itCurrentNode->GetValue(NODAL_MASS);

double& r_current_water_pressure = itCurrentNode->FastGetSolutionStepValue(WATER_PRESSURE);
double& r_current_dt_water_pressure = itCurrentNode->FastGetSolutionStepValue(DT_WATER_PRESSURE);

const array_1d<double, 3>& r_external_force = itCurrentNode->FastGetSolutionStepValue(EXTERNAL_FORCE);
const array_1d<double, 3>& r_internal_force = itCurrentNode->FastGetSolutionStepValue(INTERNAL_FORCE);
const array_1d<double, 3>& r_damping_force = itCurrentNode->FastGetSolutionStepValue(DAMPING_FORCE);

std::array<bool, 3> fix_displacements = {false, false, false};
fix_displacements[0] = (itCurrentNode->GetDof(DISPLACEMENT_X, DisplacementPosition).IsFixed());
fix_displacements[1] = (itCurrentNode->GetDof(DISPLACEMENT_Y, DisplacementPosition + 1).IsFixed());
if (DomainSize == 3)
fix_displacements[2] = (itCurrentNode->GetDof(DISPLACEMENT_Z, DisplacementPosition + 2).IsFixed());

if (nodal_mass > numerical_limit){
for (IndexType j = 0; j < DomainSize; j++) {
if (fix_displacements[j] == false) {
r_velocity[j] += 0.5 * mDeltaTime * (r_external_force[j]
- r_internal_force[j]
- r_damping_force[j])/nodal_mass;
}
}
}
else {
noalias(r_velocity) = ZeroVector(3);
}
if( itCurrentNode->IsFixed(WATER_PRESSURE) == false ) {
r_current_water_pressure = 0.0;
r_current_dt_water_pressure = 0.0;
}

const array_1d<double, 3>& r_velocity_old = itCurrentNode->FastGetSolutionStepValue(VELOCITY,1);
array_1d<double, 3>& r_acceleration = itCurrentNode->FastGetSolutionStepValue(ACCELERATION);

noalias(r_acceleration) = (1.0/mDeltaTime) * (r_velocity - r_velocity_old);
}


void CalculateRHSContribution(
Element& rCurrentElement,
LocalSystemVectorType& RHS_Contribution,
Element::EquationIdVectorType& EquationId,
const ProcessInfo& rCurrentProcessInfo
) override
{
KRATOS_TRY

rCurrentElement.AddExplicitContribution(RHS_Contribution, RESIDUAL_VECTOR, DAMPING_FORCE, rCurrentProcessInfo);

KRATOS_CATCH("")
}






protected:












private:








}; 




} 

#endif 
