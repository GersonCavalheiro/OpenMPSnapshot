
#pragma once



#include "solving_strategies/schemes/scheme.h"
#include "utilities/variable_utils.h"
#include "custom_utilities/explicit_integration_utilities.h"

namespace Kratos {








template <class TSparseSpace,
class TDenseSpace 
>
class ExplicitMultiStageKimScheme
: public Scheme<TSparseSpace, TDenseSpace> {

public:

typedef Scheme<TSparseSpace, TDenseSpace> BaseType;

typedef typename BaseType::DofsArrayType DofsArrayType;
typedef typename BaseType::TSystemMatrixType TSystemMatrixType;
typedef typename BaseType::TSystemVectorType TSystemVectorType;
typedef typename BaseType::LocalSystemVectorType LocalSystemVectorType;

typedef ModelPart::ElementsContainerType ElementsArrayType;
typedef ModelPart::NodesContainerType NodesArrayType;

typedef std::size_t SizeType;

typedef std::size_t IndexType;

typedef typename ModelPart::NodeIterator NodeIterator;

static constexpr double numerical_limit = std::numeric_limits<double>::epsilon();

typedef Variable<array_1d<double,3>> ArrayVarType;
typedef Variable<double> DoubleVarType;

typedef array_1d<double, 3> Double3DArray;

KRATOS_CLASS_POINTER_DEFINITION(ExplicitMultiStageKimScheme);



ExplicitMultiStageKimScheme(
const double DeltaTimeFraction
)
: Scheme<TSparseSpace, TDenseSpace>()
{
mDeltaTime.Fraction = DeltaTimeFraction;
}


ExplicitMultiStageKimScheme(Parameters rParameters =  Parameters(R"({})"))
: Scheme<TSparseSpace, TDenseSpace>()
{
Parameters default_parameters = Parameters(R"(
{
"fraction_delta_time"        : 0.333333333333333333333333333333333333
})" );

rParameters.ValidateAndAssignDefaults(default_parameters);
mDeltaTime.Fraction = rParameters["fraction_delta_time"].GetDouble();
}


virtual ~ExplicitMultiStageKimScheme() {}



int Check(const ModelPart& rModelPart) const override
{
KRATOS_TRY;

BaseType::Check(rModelPart);

KRATOS_ERROR_IF(rModelPart.GetBufferSize() < 2) << "Insufficient buffer size for Central Difference Scheme. It has to be > 2" << std::endl;

KRATOS_ERROR_IF_NOT(rModelPart.GetProcessInfo().Has(DOMAIN_SIZE)) << "DOMAIN_SIZE not defined on ProcessInfo. Please define" << std::endl;

KRATOS_ERROR_IF((mDeltaTime.Fraction <= 0.0) || (mDeltaTime.Fraction >= 1.0)) << "fraction_delta_time must be >0 && <1 !" << std::endl;

KRATOS_ERROR_IF_NOT(rModelPart.NumberOfNodes()>0) << "model part contains 0 nodes" << std::endl;

return 0;

KRATOS_CATCH("");
}


void Initialize(ModelPart& rModelPart) override
{
KRATOS_TRY

const ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();

mTime.Delta = r_current_process_info[DELTA_TIME];
mTime.MidStep = mTime.Delta * mDeltaTime.Fraction;

const SizeType dim = r_current_process_info[DOMAIN_SIZE];

if (!BaseType::SchemeIsInitialized())
InitializeExplicitScheme(rModelPart, dim);
else
SchemeCustomInitialization(rModelPart, dim);

BaseType::SetSchemeIsInitialized();

KRATOS_CATCH("")
}


void InitializeSolutionStep(
ModelPart& rModelPart,
TSystemMatrixType& rA,
TSystemVectorType& rDx,
TSystemVectorType& rb
) override
{
KRATOS_TRY
BaseType::InitializeSolutionStep(rModelPart, rA, rDx, rb);
InitializeResidual(rModelPart);
KRATOS_CATCH("")
}


void InitializeResidual(ModelPart& rModelPart)
{
KRATOS_TRY

NodesArrayType& r_nodes = rModelPart.Nodes();

const Double3DArray zero_array = ZeroVector(3);
VariableUtils().SetVariable(FORCE_RESIDUAL, zero_array,r_nodes);
const bool has_dof_for_rot_z = (r_nodes.begin())->HasDofFor(ROTATION_Z);
if (has_dof_for_rot_z)
VariableUtils().SetVariable(MOMENT_RESIDUAL,zero_array,r_nodes);

KRATOS_CATCH("")
}


void InitializeExplicitScheme(
ModelPart& rModelPart,
const SizeType DomainSize = 3
)
{
KRATOS_TRY

NodesArrayType& r_nodes = rModelPart.Nodes();

const auto it_node_begin = rModelPart.NodesBegin();

const Double3DArray zero_array = ZeroVector(3);
#pragma omp parallel for schedule(guided,512)
for (int i = 0; i < static_cast<int>(r_nodes.size()); ++i) {
auto it_node = (it_node_begin + i);
it_node->SetValue(NODAL_MASS, 0.0);
Double3DArray& r_fractional_acceleration = it_node->FastGetSolutionStepValue(FRACTIONAL_ACCELERATION);
r_fractional_acceleration  = ZeroVector(3);
}
const bool has_dof_for_rot_z = it_node_begin->HasDofFor(ROTATION_Z);
if (has_dof_for_rot_z) {
#pragma omp parallel for schedule(guided,512)
for (int i = 0; i < static_cast<int>(r_nodes.size()); ++i) {
auto it_node = (it_node_begin + i);
it_node->SetValue(NODAL_INERTIA, zero_array);
Double3DArray& r_fractional_acceleration = it_node->FastGetSolutionStepValue(FRACTIONAL_ANGULAR_ACCELERATION);
r_fractional_acceleration  = ZeroVector(3);
}
}

#pragma omp parallel for schedule(guided,512)
for (int i = 0; i < static_cast<int>(r_nodes.size()); ++i) {
auto it_node = (it_node_begin + i);

Double3DArray& r_current_residual = it_node->FastGetSolutionStepValue(FORCE_RESIDUAL);

for (IndexType j = 0; j < DomainSize; j++) {
r_current_residual[j] = 0.0;}

if (has_dof_for_rot_z) {
Double3DArray& r_current_residual_moment = it_node->FastGetSolutionStepValue(MOMENT_RESIDUAL);

const IndexType initial_j = DomainSize == 3 ? 0 : 2; 
for (IndexType j = initial_j; j < 3; j++) {
r_current_residual_moment[j] = 0.0;}
}
}

KRATOS_CATCH("")
}


void Update(
ModelPart& rModelPart,
DofsArrayType& rDofSet,
TSystemMatrixType& rA,
TSystemVectorType& rDx,
TSystemVectorType& rb
) override
{
KRATOS_TRY

const ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();

NodesArrayType& r_nodes = rModelPart.Nodes();

const SizeType dim = r_current_process_info[DOMAIN_SIZE];

const auto it_node_begin = rModelPart.NodesBegin();
const bool has_dof_for_rot_z = it_node_begin->HasDofFor(ROTATION_Z);


const IndexType disppos = it_node_begin->GetDofPosition(DISPLACEMENT_X);
const IndexType rotppos = has_dof_for_rot_z ? it_node_begin->GetDofPosition(ROTATION_X) : 0;


#pragma omp parallel for schedule(guided,512)
for (int i = 0; i < static_cast<int>(r_nodes.size()); ++i) {
this->UpdateAccelerationStage(
it_node_begin + i, disppos,ACCELERATION,VELOCITY,
NODAL_DISPLACEMENT_DAMPING,FORCE_RESIDUAL,NODAL_MASS,
DISPLACEMENT_X,DISPLACEMENT_Y,DISPLACEMENT_Z,dim);
} 

if (has_dof_for_rot_z){
#pragma omp parallel for schedule(guided,512)
for (int i = 0; i < static_cast<int>(r_nodes.size()); ++i) {
this->UpdateAccelerationStage(
it_node_begin + i, rotppos,ANGULAR_ACCELERATION,
ANGULAR_VELOCITY,NODAL_ROTATION_DAMPING,MOMENT_RESIDUAL,
NODAL_INERTIA,ROTATION_X,ROTATION_Y,ROTATION_Z,dim);
} 
}


#pragma omp parallel for schedule(guided,512)
for (int i = 0; i < static_cast<int>(r_nodes.size()); ++i) {
this->UpdateDegreesOfFreedomStage3(it_node_begin + i,
VELOCITY,DISPLACEMENT,ACCELERATION,FRACTIONAL_ACCELERATION,dim);
} 

if (has_dof_for_rot_z){
#pragma omp parallel for schedule(guided,512)
for (int i = 0; i < static_cast<int>(r_nodes.size()); ++i) {
this->UpdateDegreesOfFreedomStage3(it_node_begin + i,
ANGULAR_VELOCITY,ROTATION,ANGULAR_ACCELERATION,
FRACTIONAL_ANGULAR_ACCELERATION,dim);
} 
}

KRATOS_CATCH("")
}


template <typename TObjectType>
void UpdateAccelerationStage(
NodeIterator itCurrentNode,
const IndexType FieldPosition,
const ArrayVarType& rAccelerationVariable,
const ArrayVarType& rVelocityVariable,
const TObjectType& rDampingVariable,
const ArrayVarType& rResidualVariable,
const TObjectType& rIntertiaVariable,
const DoubleVarType& rFixVariable1,
const DoubleVarType& rFixVariable2,
const DoubleVarType& rFixVariable3,
const SizeType DomainSize = 3
)
{
Double3DArray& r_acceleration = itCurrentNode->FastGetSolutionStepValue(rAccelerationVariable);
Double3DArray& r_current_velocity = itCurrentNode->FastGetSolutionStepValue(rVelocityVariable);
const Double3DArray& r_current_residual = itCurrentNode->FastGetSolutionStepValue(rResidualVariable);

if (std::is_same<TObjectType,DoubleVarType>::value) {
const double& r_nodal_inertia = itCurrentNode->GetValue(NODAL_MASS);
const double& r_nodal_damping = itCurrentNode->GetValue(NODAL_DISPLACEMENT_DAMPING);

if (r_nodal_inertia > numerical_limit)
{
noalias(r_acceleration) = (r_current_residual - r_nodal_damping * r_current_velocity) / r_nodal_inertia;
}
else
noalias(r_acceleration) = ZeroVector(3);
}
else if (std::is_same<TObjectType,ArrayVarType>::value) {
const Double3DArray& r_nodal_inertia = itCurrentNode->GetValue(NODAL_INERTIA);
const Double3DArray& r_nodal_rotational_damping = itCurrentNode->GetValue(NODAL_ROTATION_DAMPING);

const IndexType initial_k = DomainSize == 3 ? 0 : 2; 
for (IndexType kk = initial_k; kk < 3; ++kk) {
if (r_nodal_inertia[kk] > numerical_limit)
r_acceleration[kk] = (r_current_residual[kk] - r_nodal_rotational_damping[kk] * r_current_velocity[kk]) / r_nodal_inertia[kk];
else
r_acceleration[kk] = 0.0;
}
}
else KRATOS_ERROR << "cannot handle input variable in \"explicit_multi_stage_kim_scheme.hpp\"" << std::endl;

std::array<bool, 3> fix_field = {false, false, false};
fix_field[0] = (itCurrentNode->GetDof(rFixVariable1, FieldPosition).IsFixed());
fix_field[1] = (itCurrentNode->GetDof(rFixVariable2, FieldPosition + 1).IsFixed());
if (DomainSize == 3)
fix_field[2] = (itCurrentNode->GetDof(rFixVariable3, FieldPosition + 2).IsFixed());

for (IndexType j = 0; j < DomainSize; j++) {
if (fix_field[j]) {
r_acceleration[j] = 0.0;
}
}
}


void UpdateDegreesOfFreedomStage1(
NodeIterator itCurrentNode,
const ArrayVarType& rVelocityVariable,
const ArrayVarType& rDisplacementVariable,
const ArrayVarType& rAccelerationVariable,
const SizeType DomainSize = 3
)
{
Double3DArray& r_current_velocity = itCurrentNode->FastGetSolutionStepValue(rVelocityVariable);
Double3DArray& r_current_displacement = itCurrentNode->FastGetSolutionStepValue(rDisplacementVariable);

const Double3DArray& r_previous_displacement = itCurrentNode->FastGetSolutionStepValue(rDisplacementVariable, 1);
const Double3DArray& r_previous_velocity = itCurrentNode->FastGetSolutionStepValue(rVelocityVariable, 1);
const Double3DArray& r_previous_acceleration = itCurrentNode->FastGetSolutionStepValue(rAccelerationVariable, 1);

for (IndexType j = 0; j < DomainSize; j++) {

r_current_displacement[j] = r_previous_displacement[j] + mTime.MidStep * r_previous_velocity[j];
r_current_displacement[j] += 0.50 * mTime.MidStep * mTime.MidStep * r_previous_acceleration[j];

r_current_velocity[j] =  r_previous_velocity[j] + mTime.MidStep * r_previous_acceleration[j];
} 
}

void UpdateDegreesOfFreedomStage2(
NodeIterator itCurrentNode,
const ArrayVarType& rVelocityVariable,
const ArrayVarType& rDisplacementVariable,
const ArrayVarType& rAccelerationVariable,
const ArrayVarType& rFractionalAccelerationVariable,
const SizeType DomainSize = 3
)
{
Double3DArray& r_current_velocity = itCurrentNode->FastGetSolutionStepValue(rVelocityVariable);
Double3DArray& r_current_displacement = itCurrentNode->FastGetSolutionStepValue(rDisplacementVariable);
Double3DArray& r_fractional_acceleration = itCurrentNode->FastGetSolutionStepValue(rFractionalAccelerationVariable);

const Double3DArray& r_previous_displacement = itCurrentNode->FastGetSolutionStepValue(rDisplacementVariable, 1);
const Double3DArray& r_previous_velocity = itCurrentNode->FastGetSolutionStepValue(rVelocityVariable, 1);
const Double3DArray& r_previous_acceleration = itCurrentNode->FastGetSolutionStepValue(rAccelerationVariable, 1);


for (IndexType j = 0; j < DomainSize; j++) {

r_current_displacement[j] = r_previous_displacement[j] + mTime.Delta * r_previous_velocity[j];
r_current_displacement[j] += 0.50 * mTime.Delta * mTime.Delta * r_previous_acceleration[j] * ((3.0*mDeltaTime.Fraction-1.0)/(3.0*mDeltaTime.Fraction));
r_current_displacement[j] += 0.50 * mTime.Delta * mTime.Delta * r_fractional_acceleration[j] * (1.0/(3.0*mDeltaTime.Fraction));


r_current_velocity[j] = r_previous_velocity[j];
r_current_velocity[j] += mTime.Delta * r_previous_acceleration[j] * ((2.0*mDeltaTime.Fraction-1.0)/(2.0*mDeltaTime.Fraction));
r_current_velocity[j] += mTime.Delta * r_fractional_acceleration[j] * ((1.0)/(2.0*mDeltaTime.Fraction));
} 
}

void UpdateDegreesOfFreedomStage3(
NodeIterator itCurrentNode,
const ArrayVarType& rVelocityVariable,
const ArrayVarType& rDisplacementVariable,
const ArrayVarType& rAccelerationVariable,
const ArrayVarType& rFractionalAccelerationVariable,
const SizeType DomainSize = 3
)
{
Double3DArray& r_current_velocity = itCurrentNode->FastGetSolutionStepValue(rVelocityVariable);
Double3DArray& r_current_displacement = itCurrentNode->FastGetSolutionStepValue(rDisplacementVariable);
Double3DArray& r_current_acceleration = itCurrentNode->FastGetSolutionStepValue(rAccelerationVariable);
Double3DArray& r_fractional_acceleration = itCurrentNode->FastGetSolutionStepValue(rFractionalAccelerationVariable);

const Double3DArray& r_previous_displacement = itCurrentNode->FastGetSolutionStepValue(rDisplacementVariable, 1);
const Double3DArray& r_previous_velocity = itCurrentNode->FastGetSolutionStepValue(rVelocityVariable, 1);
const Double3DArray& r_previous_acceleration = itCurrentNode->FastGetSolutionStepValue(rAccelerationVariable, 1);


for (IndexType j = 0; j < DomainSize; j++) {

r_current_displacement[j] = r_previous_displacement[j] + mTime.Delta * r_previous_velocity[j];
r_current_displacement[j] += 0.50 * mTime.Delta * mTime.Delta * r_previous_acceleration[j] * ((4.0*mDeltaTime.Fraction-1.0)/(6.0*mDeltaTime.Fraction));
r_current_displacement[j] -= 0.50 * mTime.Delta * mTime.Delta * r_fractional_acceleration[j] * (1.0/(6.0*mDeltaTime.Fraction*(mDeltaTime.Fraction-1.0)));
r_current_displacement[j] += 0.50 * mTime.Delta * mTime.Delta * r_current_acceleration[j] * ((2.0*mDeltaTime.Fraction-1.0)/(6.0*(mDeltaTime.Fraction-1.0)));


r_current_velocity[j] = r_previous_velocity[j];
r_current_velocity[j] += mTime.Delta * r_previous_acceleration[j] * ((3.0*mDeltaTime.Fraction-1.0)/(6.0*mDeltaTime.Fraction));
r_current_velocity[j] -= mTime.Delta * r_fractional_acceleration[j] * ((1.0)/(6.0*mDeltaTime.Fraction*(mDeltaTime.Fraction-1.0)));
r_current_velocity[j] += mTime.Delta * r_current_acceleration[j] * ((3.0*mDeltaTime.Fraction-2.0)/(6.0*(mDeltaTime.Fraction-1.0)));
} 
}


virtual void SchemeCustomInitialization(
ModelPart& rModelPart,
const SizeType DomainSize = 3
)
{
KRATOS_TRY

NodesArrayType& r_nodes = rModelPart.Nodes();

const auto it_node_begin = rModelPart.NodesBegin();

const Double3DArray zero_array = ZeroVector(3);

const bool has_dof_for_rot_z = it_node_begin->HasDofFor(ROTATION_Z);

const IndexType disppos = it_node_begin->GetDofPosition(DISPLACEMENT_X);
const IndexType rotppos = has_dof_for_rot_z ? it_node_begin->GetDofPosition(ROTATION_X) : 0;

#pragma omp parallel for schedule(guided,512)
for (int i = 0; i < static_cast<int>(r_nodes.size()); ++i) {
auto it_node = it_node_begin + i;

const double nodal_mass = it_node->GetValue(NODAL_MASS);
const Double3DArray& r_current_residual = it_node->FastGetSolutionStepValue(FORCE_RESIDUAL);
Double3DArray& r_current_acceleration = it_node->FastGetSolutionStepValue(ACCELERATION);

if (nodal_mass > numerical_limit) {
r_current_acceleration = r_current_residual / nodal_mass;
} else {
r_current_acceleration = zero_array;
}

std::array<bool, 3> fix_displacements = {false, false, false};

fix_displacements[0] = (it_node->GetDof(DISPLACEMENT_X, disppos).IsFixed());
fix_displacements[1] = (it_node->GetDof(DISPLACEMENT_Y, disppos + 1).IsFixed());
if (DomainSize == 3)
fix_displacements[2] = (it_node->GetDof(DISPLACEMENT_Z, disppos + 2).IsFixed());

for (IndexType j = 0; j < DomainSize; j++) {
if (fix_displacements[j]) {
r_current_acceleration[j] = 0.0;
}
} 


if (has_dof_for_rot_z) {
const Double3DArray& nodal_inertia = it_node->GetValue(NODAL_INERTIA);
const Double3DArray& r_current_residual_moment = it_node->FastGetSolutionStepValue(MOMENT_RESIDUAL);
Double3DArray& r_current_angular_acceleration = it_node->FastGetSolutionStepValue(ANGULAR_ACCELERATION);

const IndexType initial_k = DomainSize == 3 ? 0 : 2; 
for (IndexType kk = initial_k; kk < 3; ++kk) {
if (nodal_inertia[kk] > numerical_limit) {
r_current_angular_acceleration[kk] = r_current_residual_moment[kk] / nodal_inertia[kk];
} else {
r_current_angular_acceleration[kk] = 0.0;
}
}

std::array<bool, 3> fix_rotation = {false, false, false};
if (DomainSize == 3) {
fix_rotation[0] = (it_node->GetDof(ROTATION_X, rotppos).IsFixed());
fix_rotation[1] = (it_node->GetDof(ROTATION_Y, rotppos + 1).IsFixed());
}
fix_rotation[2] = (it_node->GetDof(ROTATION_Z, rotppos + 2).IsFixed());

for (IndexType j = initial_k; j < 3; j++) {
if (fix_rotation[j]) {
r_current_angular_acceleration[j] = 0.0;
}
} 
}   
}     
KRATOS_CATCH("")
}


void CalculateRHSContribution(
Element& rCurrentElement,
LocalSystemVectorType& RHS_Contribution,
Element::EquationIdVectorType& EquationId,
const ProcessInfo& rCurrentProcessInfo
) override
{
KRATOS_TRY

this->TCalculateRHSContribution(rCurrentElement, RHS_Contribution, rCurrentProcessInfo);
KRATOS_CATCH("")
}


void CalculateRHSContribution(
Condition& rCurrentCondition,
LocalSystemVectorType& RHS_Contribution,
Element::EquationIdVectorType& EquationId,
const ProcessInfo& rCurrentProcessInfo
) override
{
KRATOS_TRY

this->TCalculateRHSContribution(rCurrentCondition, RHS_Contribution, rCurrentProcessInfo);

KRATOS_CATCH("")
}



void CalculateAndAddRHS(ModelPart& rModelPart)
{
KRATOS_TRY

const ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();
ConditionsArrayType& r_conditions = rModelPart.Conditions();
ElementsArrayType& r_elements = rModelPart.Elements();

LocalSystemVectorType RHS_Contribution = LocalSystemVectorType(0);
Element::EquationIdVectorType equation_id_vector_dummy; 

#pragma omp parallel for firstprivate(RHS_Contribution, equation_id_vector_dummy), schedule(guided,512)
for (int i = 0; i < static_cast<int>(r_conditions.size()); ++i) {
auto it_cond = r_conditions.begin() + i;
CalculateRHSContribution(*it_cond, RHS_Contribution, equation_id_vector_dummy, r_current_process_info);
}

#pragma omp parallel for firstprivate(RHS_Contribution, equation_id_vector_dummy), schedule(guided,512)
for (int i = 0; i < static_cast<int>(r_elements.size()); ++i) {
auto it_elem = r_elements.begin() + i;
CalculateRHSContribution(*it_elem, RHS_Contribution, equation_id_vector_dummy, r_current_process_info);
}

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
const ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();

NodesArrayType& r_nodes = rModelPart.Nodes();

const SizeType dim = r_current_process_info[DOMAIN_SIZE];

mTime.Delta = r_current_process_info[DELTA_TIME];
mTime.MidStep = mTime.Delta * mDeltaTime.Fraction;

const auto it_node_begin = rModelPart.NodesBegin();
const bool has_dof_for_rot_z = it_node_begin->HasDofFor(ROTATION_Z);


const IndexType disppos = it_node_begin->GetDofPosition(DISPLACEMENT_X);
const IndexType rotppos = has_dof_for_rot_z ? it_node_begin->GetDofPosition(ROTATION_X) : 0;


#pragma omp parallel for schedule(guided,512)
for (int i = 0; i < static_cast<int>(r_nodes.size()); ++i) {
this->UpdateDegreesOfFreedomStage1(it_node_begin + i,
VELOCITY,DISPLACEMENT,ACCELERATION,dim);
} 


if (has_dof_for_rot_z){
#pragma omp parallel for schedule(guided,512)
for (int i = 0; i < static_cast<int>(r_nodes.size()); ++i) {
this->UpdateDegreesOfFreedomStage1(it_node_begin + i,
ANGULAR_VELOCITY,ROTATION,ANGULAR_ACCELERATION,dim);
} 
}


InitializeResidual(rModelPart);
CalculateAndAddRHS(rModelPart);

#pragma omp parallel for schedule(guided,512)
for (int i = 0; i < static_cast<int>(r_nodes.size()); ++i) {
this->UpdateAccelerationStage(
it_node_begin + i, disppos,FRACTIONAL_ACCELERATION,
VELOCITY,NODAL_DISPLACEMENT_DAMPING,FORCE_RESIDUAL,NODAL_MASS,
DISPLACEMENT_X,DISPLACEMENT_Y,DISPLACEMENT_Z,dim);
} 

if (has_dof_for_rot_z){
#pragma omp parallel for schedule(guided,512)
for (int i = 0; i < static_cast<int>(r_nodes.size()); ++i) {
this->UpdateAccelerationStage(
it_node_begin + i, rotppos,FRACTIONAL_ANGULAR_ACCELERATION,
ANGULAR_VELOCITY,NODAL_ROTATION_DAMPING,MOMENT_RESIDUAL,
NODAL_INERTIA,ROTATION_X,ROTATION_Y,ROTATION_Z,dim);
} 
}


#pragma omp parallel for schedule(guided,512)
for (int i = 0; i < static_cast<int>(r_nodes.size()); ++i) {
this->UpdateDegreesOfFreedomStage2(it_node_begin + i,
VELOCITY,DISPLACEMENT,ACCELERATION,FRACTIONAL_ACCELERATION,dim);
} 

if (has_dof_for_rot_z){
#pragma omp parallel for schedule(guided,512)
for (int i = 0; i < static_cast<int>(r_nodes.size()); ++i) {
this->UpdateDegreesOfFreedomStage2(it_node_begin + i,
ANGULAR_VELOCITY,ROTATION,ANGULAR_ACCELERATION,
FRACTIONAL_ANGULAR_ACCELERATION,dim);
} 
}

InitializeResidual(rModelPart);
CalculateAndAddRHS(rModelPart);

KRATOS_CATCH("")
}






protected:



struct DeltaTimeParameters {
double Fraction;        
};


struct TimeVariables {
double MidStep;         
double Delta;          
};


TimeVariables mTime;            
DeltaTimeParameters mDeltaTime; 








private:





template <typename TObjectType>
void TCalculateRHSContribution(
TObjectType& rCurrentEntity,
LocalSystemVectorType& RHS_Contribution,
const ProcessInfo& rCurrentProcessInfo
)
{
rCurrentEntity.CalculateRightHandSide(RHS_Contribution, rCurrentProcessInfo);

rCurrentEntity.AddExplicitContribution(RHS_Contribution, RESIDUAL_VECTOR, FORCE_RESIDUAL, rCurrentProcessInfo);
rCurrentEntity.AddExplicitContribution(RHS_Contribution, RESIDUAL_VECTOR, MOMENT_RESIDUAL, rCurrentProcessInfo);
}





}; 




} 
