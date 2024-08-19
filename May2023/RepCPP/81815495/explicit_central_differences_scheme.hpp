
#pragma once



#include "solving_strategies/schemes/scheme.h"
#include "utilities/variable_utils.h"
#include "custom_utilities/explicit_integration_utilities.h"
#include "utilities/parallel_utilities.h"

namespace Kratos {








template <class TSparseSpace,
class TDenseSpace 
>
class ExplicitCentralDifferencesScheme
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

KRATOS_CLASS_POINTER_DEFINITION(ExplicitCentralDifferencesScheme);



ExplicitCentralDifferencesScheme(
const double MaximumDeltaTime,
const double DeltaTimeFraction,
const double DeltaTimePredictionLevel
)
: Scheme<TSparseSpace, TDenseSpace>()
{
mDeltaTime.PredictionLevel = DeltaTimePredictionLevel;
mDeltaTime.Maximum = MaximumDeltaTime;
mDeltaTime.Fraction = DeltaTimeFraction;
}


ExplicitCentralDifferencesScheme(Parameters rParameters =  Parameters(R"({})"))
: Scheme<TSparseSpace, TDenseSpace>()
{
Parameters default_parameters = Parameters(R"(
{
"time_step_prediction_level" : 0.0,
"fraction_delta_time"        : 0.9,
"max_delta_time"             : 1.0e0
})" );

rParameters.ValidateAndAssignDefaults(default_parameters);

mDeltaTime.PredictionLevel = rParameters["time_step_prediction_level"].GetDouble();
mDeltaTime.Maximum = rParameters["max_delta_time"].GetDouble();
mDeltaTime.Fraction = rParameters["fraction_delta_time"].GetDouble();
}


virtual ~ExplicitCentralDifferencesScheme() {}



int Check(const ModelPart& rModelPart) const override
{
KRATOS_TRY;

BaseType::Check(rModelPart);

KRATOS_ERROR_IF(rModelPart.GetBufferSize() < 2) << "Insufficient buffer size for Central Difference Scheme. It has to be > 2" << std::endl;

KRATOS_ERROR_IF_NOT(rModelPart.GetProcessInfo().Has(DOMAIN_SIZE)) << "DOMAIN_SIZE not defined on ProcessInfo. Please define" << std::endl;

return 0;

KRATOS_CATCH("");
}


void Initialize(ModelPart& rModelPart) override
{
KRATOS_TRY

if ((mDeltaTime.PredictionLevel > 0) && (!BaseType::SchemeIsInitialized())) {
Parameters prediction_parameters = Parameters(R"(
{
"time_step_prediction_level" : 2.0,
"max_delta_time"             : 1.0e0,
"safety_factor"              : 0.8
})" );
prediction_parameters["time_step_prediction_level"].SetDouble(mDeltaTime.PredictionLevel);
prediction_parameters["max_delta_time"].SetDouble(mDeltaTime.Maximum);
ExplicitIntegrationUtilities::CalculateDeltaTime(rModelPart, prediction_parameters);
}

const ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();

mTime.Current = r_current_process_info[TIME] + r_current_process_info[DELTA_TIME];
mTime.Delta = r_current_process_info[DELTA_TIME];
mTime.Middle = mTime.Current - 0.5 * mTime.Delta;
mTime.Previous = mTime.Current - mTime.Delta;
mTime.PreviousMiddle = mTime.Current - 1.5 * mTime.Delta;

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
if (mDeltaTime.PredictionLevel > 1) {
Parameters prediction_parameters = Parameters(R"(
{
"time_step_prediction_level" : 2.0,
"max_delta_time"             : 1.0e0,
"safety_factor"              : 0.8
})" );
prediction_parameters["time_step_prediction_level"].SetDouble(mDeltaTime.PredictionLevel); 
prediction_parameters["max_delta_time"].SetDouble(mDeltaTime.Maximum);
ExplicitIntegrationUtilities::CalculateDeltaTime(rModelPart, prediction_parameters);
}
InitializeResidual(rModelPart);
KRATOS_CATCH("")
}


void InitializeResidual(ModelPart& rModelPart)
{
KRATOS_TRY

NodesArrayType& r_nodes = rModelPart.Nodes();

const array_1d<double, 3> zero_array = ZeroVector(3);
VariableUtils().SetVariable(FORCE_RESIDUAL, zero_array,r_nodes);
const bool has_dof_for_rot_z = !r_nodes.empty() && r_nodes.begin()->HasDofFor(ROTATION_Z);
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

if (!r_nodes.empty()) {
const auto it_node_begin = rModelPart.NodesBegin();

std::function<void(Node&)> initializer_base, initializer;
const array_1d<double, 3> zero_array = ZeroVector(3);

initializer_base = [&zero_array, DomainSize](Node& rNode){
rNode.SetValue(NODAL_MASS, 0.0);
rNode.FastGetSolutionStepValue(MIDDLE_VELOCITY) = zero_array;

array_1d<double, 3>& r_middle_velocity = rNode.FastGetSolutionStepValue(MIDDLE_VELOCITY);
const array_1d<double, 3>& r_current_velocity = rNode.FastGetSolutionStepValue(VELOCITY);
array_1d<double, 3>& r_current_residual = rNode.FastGetSolutionStepValue(FORCE_RESIDUAL);

for (IndexType j = 0; j < DomainSize; j++) {
r_middle_velocity[j] = r_current_velocity[j];
r_current_residual[j] = 0.0;
}
};

const bool has_dof_for_rot_z = it_node_begin->HasDofFor(ROTATION_Z);
if (has_dof_for_rot_z) {
initializer = [&zero_array, DomainSize, &initializer_base](Node& rNode){
initializer_base(rNode);
rNode.SetValue(NODAL_INERTIA, zero_array);
rNode.FastGetSolutionStepValue(MIDDLE_ANGULAR_VELOCITY) = zero_array;

array_1d<double, 3>& r_middle_angular_velocity = rNode.FastGetSolutionStepValue(MIDDLE_ANGULAR_VELOCITY);
const array_1d<double, 3>& r_current_angular_velocity = rNode.FastGetSolutionStepValue(ANGULAR_VELOCITY);
array_1d<double, 3>& r_current_residual_moment = rNode.FastGetSolutionStepValue(MOMENT_RESIDUAL);

const IndexType initial_j = DomainSize == 3 ? 0 : 2; 
for (IndexType j = initial_j; j < 3; j++) {
r_middle_angular_velocity[j] = r_current_angular_velocity[j];
r_current_residual_moment[j] = 0.0;
}
};
} else {
initializer = initializer_base;
}

block_for_each(r_nodes, initializer);
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
NodesArrayType& r_nodes = rModelPart.Nodes();

if (!r_nodes.empty()) {
const ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();

const SizeType dim = r_current_process_info[DOMAIN_SIZE];

mTime.Current = r_current_process_info[TIME];
mTime.Delta = r_current_process_info[DELTA_TIME];

mTime.Middle   = mTime.Current - 0.50*mTime.Delta;
mTime.Previous = mTime.Current - 1.00*mTime.Delta;
mTime.PreviousMiddle = mTime.Middle - 1.00*mTime.Delta;

if (mTime.Previous<0.0) mTime.Previous=0.00;
if (mTime.PreviousMiddle<0.0) mTime.PreviousMiddle=0.00;
const auto it_node_begin = rModelPart.NodesBegin();
const bool has_dof_for_rot_z = it_node_begin->HasDofFor(ROTATION_Z);

const IndexType disppos = it_node_begin->GetDofPosition(DISPLACEMENT_X);
const IndexType rotppos = has_dof_for_rot_z ? it_node_begin->GetDofPosition(ROTATION_X) : 0;

std::function<void(SizeType)> updater_base, updater;
updater_base = [it_node_begin, dim, &disppos, this](SizeType index){
this->UpdateTranslationalDegreesOfFreedom(it_node_begin + index, disppos, dim);
};

if (has_dof_for_rot_z) {
updater = [it_node_begin, dim, &rotppos, this, &updater_base](SizeType index){
updater_base(index);
this->UpdateRotationalDegreesOfFreedom(it_node_begin + index, rotppos, dim);
};
} else {
updater = updater_base;
}

IndexPartition<SizeType>(r_nodes.size()).for_each(updater);
} 

KRATOS_CATCH("")
}


void UpdateTranslationalDegreesOfFreedom(
NodeIterator itCurrentNode,
const IndexType DisplacementPosition,
const SizeType DomainSize = 3
)
{

const double nodal_mass = itCurrentNode->GetValue(NODAL_MASS);
const double nodal_displacement_damping = itCurrentNode->GetValue(NODAL_DISPLACEMENT_DAMPING);
const array_1d<double, 3>& r_current_residual = itCurrentNode->FastGetSolutionStepValue(FORCE_RESIDUAL);


array_1d<double, 3>& r_current_velocity = itCurrentNode->FastGetSolutionStepValue(VELOCITY);
array_1d<double, 3>& r_current_displacement = itCurrentNode->FastGetSolutionStepValue(DISPLACEMENT);
array_1d<double, 3>& r_middle_velocity = itCurrentNode->FastGetSolutionStepValue(MIDDLE_VELOCITY);

array_1d<double, 3>& r_current_acceleration = itCurrentNode->FastGetSolutionStepValue(ACCELERATION);

const array_1d<double, 3>& r_previous_displacement = itCurrentNode->FastGetSolutionStepValue(DISPLACEMENT, 1);
const array_1d<double, 3>& r_previous_middle_velocity = itCurrentNode->FastGetSolutionStepValue(MIDDLE_VELOCITY, 1);
if (nodal_mass > numerical_limit)
noalias(r_current_acceleration) = (r_current_residual - nodal_displacement_damping * r_current_velocity) / nodal_mass;
else
noalias(r_current_acceleration) = ZeroVector(3);

std::array<bool, 3> fix_displacements = {false, false, false};

fix_displacements[0] = (itCurrentNode->GetDof(DISPLACEMENT_X, DisplacementPosition).IsFixed());
fix_displacements[1] = (itCurrentNode->GetDof(DISPLACEMENT_Y, DisplacementPosition + 1).IsFixed());
if (DomainSize == 3)
fix_displacements[2] = (itCurrentNode->GetDof(DISPLACEMENT_Z, DisplacementPosition + 2).IsFixed());

for (IndexType j = 0; j < DomainSize; j++) {
if (fix_displacements[j]) {
r_current_acceleration[j] = 0.0;
r_middle_velocity[j] = 0.0;
}

r_current_velocity[j] =  r_previous_middle_velocity[j] + (mTime.Previous - mTime.PreviousMiddle) * r_current_acceleration[j]; 
r_middle_velocity[j] = r_current_velocity[j] + (mTime.Middle - mTime.Previous) * r_current_acceleration[j];
r_current_displacement[j] = r_previous_displacement[j] + mTime.Delta * r_middle_velocity[j];
} 
}


void UpdateRotationalDegreesOfFreedom(
NodeIterator itCurrentNode,
const IndexType RotationPosition,
const SizeType DomainSize = 3
)
{
const array_1d<double, 3>& nodal_inertia = itCurrentNode->GetValue(NODAL_INERTIA);
const array_1d<double, 3>& nodal_rotational_damping = itCurrentNode->GetValue(NODAL_ROTATION_DAMPING);
const array_1d<double, 3>& r_current_residual_moment = itCurrentNode->FastGetSolutionStepValue(MOMENT_RESIDUAL);
array_1d<double, 3>& r_current_angular_velocity = itCurrentNode->FastGetSolutionStepValue(ANGULAR_VELOCITY);
array_1d<double, 3>& r_current_rotation = itCurrentNode->FastGetSolutionStepValue(ROTATION);
array_1d<double, 3>& r_middle_angular_velocity = itCurrentNode->FastGetSolutionStepValue(MIDDLE_ANGULAR_VELOCITY);
array_1d<double, 3>& r_current_angular_acceleration = itCurrentNode->FastGetSolutionStepValue(ANGULAR_ACCELERATION);


const array_1d<double, 3>& r_previous_rotation = itCurrentNode->FastGetSolutionStepValue(ROTATION, 1);
const array_1d<double, 3>& r_previous_middle_angular_velocity = itCurrentNode->FastGetSolutionStepValue(MIDDLE_ANGULAR_VELOCITY, 1);

const IndexType initial_k = DomainSize == 3 ? 0 : 2; 
for (IndexType kk = initial_k; kk < 3; ++kk) {
if (nodal_inertia[kk] > numerical_limit)
r_current_angular_acceleration[kk] = (r_current_residual_moment[kk] - nodal_rotational_damping[kk] * r_current_angular_velocity[kk]) / nodal_inertia[kk];
else
r_current_angular_acceleration[kk] = 0.0;
}

std::array<bool, 3> fix_rotation = {false, false, false};
if (DomainSize == 3) {
fix_rotation[0] = (itCurrentNode->GetDof(ROTATION_X, RotationPosition).IsFixed());
fix_rotation[1] = (itCurrentNode->GetDof(ROTATION_Y, RotationPosition + 1).IsFixed());
}
fix_rotation[2] = (itCurrentNode->GetDof(ROTATION_Z, RotationPosition + 2).IsFixed());

for (IndexType j = initial_k; j < 3; j++) {
if (fix_rotation[j]) {
r_current_angular_acceleration[j] = 0.0;
r_middle_angular_velocity[j] = 0.0;
}
r_current_angular_velocity[j] = r_previous_middle_angular_velocity[j] + (mTime.Previous - mTime.PreviousMiddle) * r_current_angular_acceleration[j];
r_middle_angular_velocity[j] = r_current_angular_velocity[j] + (mTime.Middle - mTime.Previous) * r_current_angular_acceleration[j];
r_current_rotation[j] = r_previous_rotation[j] + mTime.Delta * r_middle_angular_velocity[j];
}
}


virtual void SchemeCustomInitialization(
ModelPart& rModelPart,
const SizeType DomainSize = 3
)
{
KRATOS_TRY

NodesArrayType& r_nodes = rModelPart.Nodes();

if (!r_nodes.empty()) {
const auto it_node_begin = rModelPart.NodesBegin();

const bool has_dof_for_rot_z = it_node_begin->HasDofFor(ROTATION_Z);

const array_1d<double, 3> zero_array = ZeroVector(3);

const IndexType disppos = it_node_begin->GetDofPosition(DISPLACEMENT_X);
const IndexType rotppos = has_dof_for_rot_z ? it_node_begin->GetDofPosition(ROTATION_X) : 0;

std::function<void(Node&)> initializer_base, initializer;

initializer_base = [&zero_array, &disppos, DomainSize, this](Node& rNode) {
const double nodal_mass = rNode.GetValue(NODAL_MASS);
const array_1d<double, 3>& r_current_residual = rNode.FastGetSolutionStepValue(FORCE_RESIDUAL);

array_1d<double, 3>& r_current_velocity = rNode.FastGetSolutionStepValue(VELOCITY);
array_1d<double, 3>& r_middle_velocity = rNode.FastGetSolutionStepValue(MIDDLE_VELOCITY);

array_1d<double, 3>& r_current_acceleration = rNode.FastGetSolutionStepValue(ACCELERATION);

if (nodal_mass > numerical_limit) {
r_current_acceleration = r_current_residual / nodal_mass;
} else {
r_current_acceleration = zero_array;
}

std::array<bool, 3> fix_displacements = {false, false, false};

fix_displacements[0] = (rNode.GetDof(DISPLACEMENT_X, disppos).IsFixed());
fix_displacements[1] = (rNode.GetDof(DISPLACEMENT_Y, disppos + 1).IsFixed());
if (DomainSize == 3)
fix_displacements[2] = (rNode.GetDof(DISPLACEMENT_Z, disppos + 2).IsFixed());

for (IndexType j = 0; j < DomainSize; j++) {
if (fix_displacements[j]) {
r_current_acceleration[j] = 0.0;
r_middle_velocity[j] = 0.0;
}

r_middle_velocity[j] = 0.0 + (mTime.Middle - mTime.Previous) * r_current_acceleration[j];
r_current_velocity[j] = r_middle_velocity[j] + (mTime.Previous - mTime.PreviousMiddle) * r_current_acceleration[j]; 

} 
};

if (has_dof_for_rot_z) {
initializer = [&rotppos, &initializer_base, DomainSize, this](Node& rNode) {
initializer_base(rNode);
const array_1d<double, 3>& nodal_inertia = rNode.GetValue(NODAL_INERTIA);
const array_1d<double, 3>& r_current_residual_moment = rNode.FastGetSolutionStepValue(MOMENT_RESIDUAL);
array_1d<double, 3>& r_current_angular_velocity = rNode.FastGetSolutionStepValue(ANGULAR_VELOCITY);
array_1d<double, 3>& r_middle_angular_velocity = rNode.FastGetSolutionStepValue(MIDDLE_ANGULAR_VELOCITY);
array_1d<double, 3>& r_current_angular_acceleration = rNode.FastGetSolutionStepValue(ANGULAR_ACCELERATION);

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
fix_rotation[0] = (rNode.GetDof(ROTATION_X, rotppos).IsFixed());
fix_rotation[1] = (rNode.GetDof(ROTATION_Y, rotppos + 1).IsFixed());
}
fix_rotation[2] = (rNode.GetDof(ROTATION_Z, rotppos + 2).IsFixed());

for (IndexType j = initial_k; j < 3; j++) {
if (fix_rotation[j]) {
r_current_angular_acceleration[j] = 0.0;
r_middle_angular_velocity[j] = 0.0;
}

r_middle_angular_velocity[j] = 0.0 +  (mTime.Middle - mTime.Previous) * r_current_angular_acceleration[j];
r_current_angular_velocity[j] = r_middle_angular_velocity[j] +  (mTime.Previous - mTime.PreviousMiddle) *  r_current_angular_acceleration[j];
}
};
} else {
initializer = initializer_base;
}

block_for_each(r_nodes, initializer);
} 

mTime.Previous = mTime.Current;
mTime.PreviousMiddle = mTime.Middle;
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

using TLS = std::tuple<LocalSystemVectorType,Element::EquationIdVectorType>;
TLS thread_local_storage; 

block_for_each(r_conditions, thread_local_storage, [&r_current_process_info, this](Condition& r_condition, TLS& r_rhs_contrib_and_equation_ids){
CalculateRHSContribution(r_condition,
std::get<0>(r_rhs_contrib_and_equation_ids),
std::get<1>(r_rhs_contrib_and_equation_ids),
r_current_process_info);
});

block_for_each(r_elements, thread_local_storage, [&r_current_process_info, this](Element& r_element, TLS& r_rhs_contrib_and_equation_ids){
CalculateRHSContribution(r_element,
std::get<0>(r_rhs_contrib_and_equation_ids),
std::get<1>(r_rhs_contrib_and_equation_ids),
r_current_process_info);
});

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
CalculateAndAddRHS(rModelPart);
KRATOS_CATCH("")
}






protected:



struct DeltaTimeParameters {
double PredictionLevel; 
double Maximum;         
double Fraction;        
};


struct TimeVariables {
double PreviousMiddle; 
double Previous;       
double Middle;         
double Current;        

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
