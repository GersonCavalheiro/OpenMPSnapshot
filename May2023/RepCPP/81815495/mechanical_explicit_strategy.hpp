
#pragma once



#include "solving_strategies/strategies/implicit_solving_strategy.h"
#include "structural_mechanics_application_variables.h"
#include "utilities/variable_utils.h"
#include "utilities/constraint_utilities.h"
#include "utilities/parallel_utilities.h"
#include "includes/ublas_interface.h"
#include "includes/variables.h"

namespace Kratos {

template <class TSparseSpace,
class TDenseSpace,  
class TLinearSolver 
>
class MechanicalExplicitStrategy
: public ImplicitSolvingStrategy<TSparseSpace, TDenseSpace, TLinearSolver> {
public:

typedef ImplicitSolvingStrategy<TSparseSpace, TDenseSpace, TLinearSolver> BaseType;

typedef typename BaseType::TSchemeType TSchemeType;
typedef typename BaseType::DofsArrayType DofsArrayType;
typedef typename BaseType::TSystemMatrixType TSystemMatrixType;
typedef typename BaseType::TSystemVectorType TSystemVectorType;
typedef typename BaseType::TSystemMatrixPointerType TSystemMatrixPointerType;
typedef typename BaseType::TSystemVectorPointerType TSystemVectorPointerType;
typedef typename BaseType::NodesArrayType NodesArrayType;
typedef typename BaseType::ElementsArrayType ElementsArrayType;
typedef typename BaseType::ConditionsArrayType ConditionsArrayType;
typedef typename BaseType::LocalSystemVectorType LocalSystemVectorType;

typedef typename Node::DofType DofType;
typedef typename DofType::Pointer DofPointerType;

KRATOS_CLASS_POINTER_DEFINITION(MechanicalExplicitStrategy);



MechanicalExplicitStrategy(
ModelPart& rModelPart,
typename TSchemeType::Pointer pScheme,
bool CalculateReactions = false,
bool ReformDofSetAtEachStep = false,
bool MoveMeshFlag = true)
: ImplicitSolvingStrategy<TSparseSpace, TDenseSpace, TLinearSolver>(rModelPart, MoveMeshFlag),
mpScheme(pScheme),
mReformDofSetAtEachStep(ReformDofSetAtEachStep),
mCalculateReactionsFlag(CalculateReactions)
{
KRATOS_TRY

BaseType::SetEchoLevel(1);

BaseType::SetRebuildLevel(0);

KRATOS_CATCH("")
}


virtual ~MechanicalExplicitStrategy()
{
Clear();
}




void SetScheme(typename TSchemeType::Pointer pScheme)
{
mpScheme = pScheme;
};


typename TSchemeType::Pointer GetScheme()
{
return mpScheme;
};



void SetInitializePerformedFlag(bool InitializePerformedFlag = true)
{
mInitializeWasPerformed = InitializePerformedFlag;
}


bool GetInitializePerformedFlag()
{
return mInitializeWasPerformed;
}


void SetCalculateReactionsFlag(bool CalculateReactionsFlag)
{
mCalculateReactionsFlag = CalculateReactionsFlag;
}


bool GetCalculateReactionsFlag()
{
return mCalculateReactionsFlag;
}


void SetReformDofSetAtEachStepFlag(bool Flag)
{
mReformDofSetAtEachStep = Flag;
}


bool GetReformDofSetAtEachStepFlag()
{
return mReformDofSetAtEachStep;
}

void Initialize() override
{
KRATOS_TRY

if (!this->mInitializeWasPerformed){
typename TSchemeType::Pointer pScheme = GetScheme();
ModelPart& r_model_part = BaseType::GetModelPart();

TSystemMatrixType matrix_a_dummy = TSystemMatrixType();

if (!pScheme->SchemeIsInitialized())pScheme->Initialize(r_model_part);

if (!pScheme->ElementsAreInitialized())pScheme->InitializeElements(r_model_part);

if (!pScheme->ConditionsAreInitialized())pScheme->InitializeConditions(r_model_part);

NodesArrayType& r_nodes = r_model_part.Nodes();
VariableUtils().SetNonHistoricalVariable(NODAL_MASS, 0.0, r_nodes);
VariableUtils().SetNonHistoricalVariable(NODAL_DISPLACEMENT_DAMPING, 0.0, r_nodes);

ElementsArrayType& r_elements = r_model_part.Elements();
ProcessInfo& r_current_process_info = r_model_part.GetProcessInfo();

Vector dummy_vector;
const bool has_dof_for_rot_z = !r_nodes.empty() && r_nodes.begin()->HasDofFor(ROTATION_Z);
if (has_dof_for_rot_z) {
const array_1d<double, 3> zero_array = ZeroVector(3);
VariableUtils().SetNonHistoricalVariable(NODAL_INERTIA, zero_array, r_nodes);
VariableUtils().SetNonHistoricalVariable(NODAL_ROTATION_DAMPING, zero_array, r_nodes);

block_for_each(r_elements, dummy_vector, [&r_current_process_info](Element& r_element, Vector& r_dummy_vector){
r_element.AddExplicitContribution(r_dummy_vector, RESIDUAL_VECTOR, NODAL_INERTIA, r_current_process_info);
});
} else { 
block_for_each(r_elements, dummy_vector, [&r_current_process_info](Element& r_element, Vector& r_dummy_vector){
r_element.AddExplicitContribution(r_dummy_vector, RESIDUAL_VECTOR, NODAL_MASS, r_current_process_info);
});
}

if(r_model_part.MasterSlaveConstraints().size() > 0) {
ConstraintUtilities::PreComputeExplicitConstraintMassAndInertia(r_model_part);
}

this->mInitializeWasPerformed = true;
}

KRATOS_CATCH("")
}


void InitializeSolutionStep() override {
KRATOS_TRY

typename TSchemeType::Pointer pScheme = GetScheme();
ModelPart& r_model_part = BaseType::GetModelPart();

TSystemMatrixType matrix_a_dummy = TSystemMatrixType();
TSystemVectorType rDx = TSystemVectorType();
TSystemVectorType rb = TSystemVectorType();

pScheme->InitializeSolutionStep(r_model_part, matrix_a_dummy, rDx, rb);

if (BaseType::mRebuildLevel > 0) { 
ProcessInfo& r_current_process_info = r_model_part.GetProcessInfo();
ElementsArrayType& r_elements = r_model_part.Elements();

NodesArrayType& r_nodes = r_model_part.Nodes();
VariableUtils().SetNonHistoricalVariable(NODAL_MASS, 0.0, r_nodes);
VariableUtils().SetNonHistoricalVariable(NODAL_DISPLACEMENT_DAMPING, 0.0, r_nodes);

Vector dummy_vector;
const bool has_dof_for_rot_z = r_model_part.Nodes().begin()->HasDofFor(ROTATION_Z);
if (has_dof_for_rot_z) {
const array_1d<double, 3> zero_array = ZeroVector(3);
VariableUtils().SetNonHistoricalVariable(NODAL_INERTIA, zero_array, r_nodes);
VariableUtils().SetNonHistoricalVariable(NODAL_ROTATION_DAMPING, zero_array, r_nodes);

block_for_each(r_elements, dummy_vector, [&r_current_process_info](Element& r_element, Vector& r_dummy_vector){
r_element.AddExplicitContribution(r_dummy_vector, RESIDUAL_VECTOR, NODAL_INERTIA, r_current_process_info);
});
} else { 
block_for_each(r_elements, dummy_vector, [&r_current_process_info](Element& r_element, Vector& r_dummy_vector){
r_element.AddExplicitContribution(r_dummy_vector, RESIDUAL_VECTOR, NODAL_MASS, r_current_process_info);
});
}

if(r_model_part.MasterSlaveConstraints().size() > 0) {
ConstraintUtilities::PreComputeExplicitConstraintMassAndInertia(r_model_part);
}
}

KRATOS_CATCH("")
}


bool SolveSolutionStep() override
{
typename TSchemeType::Pointer pScheme = GetScheme();
ModelPart& r_model_part = BaseType::GetModelPart();

DofsArrayType dof_set_dummy;
TSystemMatrixType rA = TSystemMatrixType();
TSystemVectorType rDx = TSystemVectorType();
TSystemVectorType rb = TSystemVectorType();

pScheme->InitializeNonLinIteration(BaseType::GetModelPart(), rA, rDx, rb);

pScheme->Predict(r_model_part, dof_set_dummy, rA, rDx, rb);

if(r_model_part.MasterSlaveConstraints().size() > 0) {
std::vector<std::string> dof_variable_names(2);
dof_variable_names[0] = "DISPLACEMENT";
dof_variable_names[1] = "ROTATION";
std::vector<std::string> residual_variable_names(2);
residual_variable_names[0] = "FORCE_RESIDUAL";
residual_variable_names[1] = "MOMENT_RESIDUAL";
ConstraintUtilities::PreComputeExplicitConstraintConstribution(r_model_part, dof_variable_names, residual_variable_names);
}

pScheme->Update(r_model_part, dof_set_dummy, rA, rDx, rb);

pScheme->FinalizeNonLinIteration(BaseType::GetModelPart(), rA, rDx, rb);

if(r_model_part.MasterSlaveConstraints().size() > 0) {
ComputeExplicitConstraintConstribution(pScheme, r_model_part);
}

if (mCalculateReactionsFlag) {
CalculateReactions(pScheme, r_model_part, rA, rDx, rb);
}

return true;
}


void FinalizeSolutionStep() override
{
typename TSchemeType::Pointer pScheme = GetScheme();
ModelPart& r_model_part = BaseType::GetModelPart();
TSystemMatrixType rA = TSystemMatrixType();
TSystemVectorType rDx = TSystemVectorType();
TSystemVectorType rb = TSystemVectorType();
pScheme->FinalizeSolutionStep(r_model_part, rA, rDx, rb);

if (BaseType::MoveMeshFlag())
BaseType::MoveMesh();

pScheme->Clean();
}


void Clear() override
{
KRATOS_TRY

KRATOS_INFO("MechanicalExplicitStrategy") << "Clear function used" << std::endl;

GetScheme()->Clear();
mInitializeWasPerformed = false;

KRATOS_CATCH("")
}


int Check() override
{
KRATOS_TRY

BaseType::Check();

GetScheme()->Check(BaseType::GetModelPart());

return 0;

KRATOS_CATCH("")
}




private:




void ComputeExplicitConstraintConstribution(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart
)
{
ConstraintUtilities::ResetSlaveDofs(rModelPart);

ConstraintUtilities::ApplyConstraints(rModelPart);
}


void CalculateReactions(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemMatrixType& rA,
TSystemVectorType& rDx,
TSystemVectorType& rb
)
{
auto& r_nodes = rModelPart.Nodes();

if (!r_nodes.empty()) {
const bool has_dof_for_rot_z = (r_nodes.begin())->HasDofFor(ROTATION_Z);

const array_1d<double,3> zero_array = ZeroVector(3);

const auto it_node_begin = r_nodes.begin();
const IndexType disppos = it_node_begin->GetDofPosition(DISPLACEMENT_X);
const IndexType rotppos = it_node_begin->GetDofPosition(ROTATION_X);

std::function<void(Node&)> loop_base, loop;

loop_base = [&disppos](Node& rNode){
const auto force_residual = rNode.FastGetSolutionStepValue(FORCE_RESIDUAL);

if (rNode.GetDof(DISPLACEMENT_X, disppos).IsFixed()) {
double& r_reaction = rNode.FastGetSolutionStepValue(REACTION_X);
r_reaction = force_residual[0];
}
if (rNode.GetDof(DISPLACEMENT_Y, disppos + 1).IsFixed()) {
double& r_reaction = rNode.FastGetSolutionStepValue(REACTION_Y);
r_reaction = force_residual[1];
}
if (rNode.GetDof(DISPLACEMENT_Z, disppos + 2).IsFixed()) {
double& r_reaction = rNode.FastGetSolutionStepValue(REACTION_Z);
r_reaction = force_residual[2];
}
};

if (has_dof_for_rot_z) {
loop = [&rotppos, &loop_base](Node& rNode){
loop_base(rNode);
const auto moment_residual = rNode.FastGetSolutionStepValue(MOMENT_RESIDUAL);
if (rNode.GetDof(ROTATION_X, rotppos).IsFixed()) {
double& r_reaction = rNode.FastGetSolutionStepValue(REACTION_MOMENT_X);
r_reaction = moment_residual[0];
}
if (rNode.GetDof(ROTATION_Y, rotppos + 1).IsFixed()) {
double& r_reaction = rNode.FastGetSolutionStepValue(REACTION_MOMENT_Y);
r_reaction = moment_residual[1];
}
if (rNode.GetDof(ROTATION_Z, rotppos + 2).IsFixed()) {
double& r_reaction = rNode.FastGetSolutionStepValue(REACTION_MOMENT_Z);
r_reaction = moment_residual[2];
}
};
} else {
loop = loop_base;
}

block_for_each(r_nodes, loop);
} 
}





protected:


typename TSchemeType::Pointer mpScheme; 


bool mReformDofSetAtEachStep = false;


bool mCalculateReactionsFlag = true;


bool mInitializeWasPerformed = false;







MechanicalExplicitStrategy(const MechanicalExplicitStrategy& Other){};


}; 


} 
