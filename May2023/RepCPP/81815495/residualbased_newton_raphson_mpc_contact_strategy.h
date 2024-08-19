
#pragma once



#include "contact_structural_mechanics_application_variables.h"
#include "includes/kratos_parameters.h"
#include "includes/define.h"
#include "includes/model_part.h"
#include "includes/variables.h"


#include "solving_strategies/strategies/residualbased_newton_raphson_strategy.h"


#include "custom_strategies/custom_convergencecriterias/mpc_contact_criteria.h"


#include "utilities/variable_utils.h"
#include "utilities/color_utilities.h"
#include "utilities/math_utils.h"
#include "utilities/atomic_utilities.h"


namespace Kratos {







template<class TSparseSpace,
class TDenseSpace, 
class TLinearSolver 
>

class ResidualBasedNewtonRaphsonMPCContactStrategy :
public ResidualBasedNewtonRaphsonStrategy< TSparseSpace, TDenseSpace, TLinearSolver >
{
public:


KRATOS_CLASS_POINTER_DEFINITION( ResidualBasedNewtonRaphsonMPCContactStrategy );

typedef SolvingStrategy<TSparseSpace, TDenseSpace>                        SolvingStrategyType;

typedef ImplicitSolvingStrategy<TSparseSpace, TDenseSpace, TLinearSolver>    StrategyBaseType;

typedef ResidualBasedNewtonRaphsonStrategy<TSparseSpace, TDenseSpace, TLinearSolver> BaseType;

typedef ResidualBasedNewtonRaphsonMPCContactStrategy<TSparseSpace, TDenseSpace, TLinearSolver> ClassType;

typedef ConvergenceCriteria<TSparseSpace, TDenseSpace>               TConvergenceCriteriaType;

typedef MPCContactCriteria<TSparseSpace, TDenseSpace>                 TMPCContactCriteriaType;

typedef typename BaseType::TBuilderAndSolverType                        TBuilderAndSolverType;

typedef typename BaseType::TDataType                                                TDataType;

typedef TSparseSpace                                                          SparseSpaceType;

typedef typename BaseType::TSchemeType                                            TSchemeType;

typedef typename BaseType::DofsArrayType                                        DofsArrayType;

typedef typename BaseType::TSystemMatrixType                                TSystemMatrixType;

typedef typename BaseType::TSystemVectorType                                TSystemVectorType;

typedef typename BaseType::LocalSystemVectorType                        LocalSystemVectorType;

typedef typename BaseType::LocalSystemMatrixType                        LocalSystemMatrixType;

typedef typename BaseType::TSystemMatrixPointerType                  TSystemMatrixPointerType;

typedef typename BaseType::TSystemVectorPointerType                  TSystemVectorPointerType;

typedef ModelPart::NodesContainerType                                          NodesArrayType;

typedef ModelPart::ElementsContainerType                                    ElementsArrayType;

typedef ModelPart::ConditionsContainerType                                ConditionsArrayType;

typedef ModelPart::MasterSlaveConstraintContainerType                     ConstraintArrayType;

typedef std::size_t                                                                 IndexType;

typedef std::size_t                                                                  SizeType;


explicit ResidualBasedNewtonRaphsonMPCContactStrategy()
{
}


explicit ResidualBasedNewtonRaphsonMPCContactStrategy(ModelPart& rModelPart, Parameters ThisParameters)
: BaseType(rModelPart)
{
ThisParameters = this->ValidateAndAssignParameters(ThisParameters, this->GetDefaultParameters());
this->AssignSettings(ThisParameters);
}


explicit ResidualBasedNewtonRaphsonMPCContactStrategy(
ModelPart& rModelPart,
typename TSchemeType::Pointer pScheme,
typename TConvergenceCriteriaType::Pointer pNewConvergenceCriteria,
typename TBuilderAndSolverType::Pointer pNewBuilderAndSolver,
IndexType MaxIterations = 30,
bool CalculateReactions = false,
bool ReformDofSetAtEachStep = false,
bool MoveMeshFlag = false,
Parameters ThisParameters =  Parameters(R"({})")
)
: BaseType(rModelPart, pScheme, pNewConvergenceCriteria, pNewBuilderAndSolver, MaxIterations, CalculateReactions, ReformDofSetAtEachStep, MoveMeshFlag ),
mThisParameters(ThisParameters)
{
KRATOS_TRY;

mpMPCContactCriteria = Kratos::make_shared<TMPCContactCriteriaType>();

Parameters default_parameters = GetDefaultParameters();
mThisParameters.ValidateAndAssignDefaults(default_parameters);

KRATOS_CATCH("");
}


explicit ResidualBasedNewtonRaphsonMPCContactStrategy(
ModelPart& rModelPart,
typename TSchemeType::Pointer pScheme,
typename TLinearSolver::Pointer pNewLinearSolver,
typename TConvergenceCriteriaType::Pointer pNewConvergenceCriteria,
IndexType MaxIterations = 30,
bool CalculateReactions = false,
bool ReformDofSetAtEachStep = false,
bool MoveMeshFlag = false,
Parameters ThisParameters =  Parameters(R"({})")
)
: BaseType(rModelPart, pScheme, pNewLinearSolver, pNewConvergenceCriteria, MaxIterations, CalculateReactions, ReformDofSetAtEachStep, MoveMeshFlag),
mThisParameters(ThisParameters)
{
KRATOS_TRY;

mpMPCContactCriteria = Kratos::make_shared<TMPCContactCriteriaType>();

Parameters default_parameters = GetDefaultParameters();
mThisParameters.ValidateAndAssignDefaults(default_parameters);

KRATOS_CATCH("");
}


explicit ResidualBasedNewtonRaphsonMPCContactStrategy(
ModelPart& rModelPart,
typename TSchemeType::Pointer pScheme,
typename TLinearSolver::Pointer pNewLinearSolver,
typename TConvergenceCriteriaType::Pointer pNewConvergenceCriteria,
typename TBuilderAndSolverType::Pointer pNewBuilderAndSolver,
IndexType MaxIterations = 30,
bool CalculateReactions = false,
bool ReformDofSetAtEachStep = false,
bool MoveMeshFlag = false,
Parameters ThisParameters =  Parameters(R"({})")
)
: BaseType(rModelPart, pScheme, pNewLinearSolver, pNewConvergenceCriteria, pNewBuilderAndSolver, MaxIterations, CalculateReactions, ReformDofSetAtEachStep, MoveMeshFlag ),
mThisParameters(ThisParameters)
{
KRATOS_TRY;

mpMPCContactCriteria = Kratos::make_shared<TMPCContactCriteriaType>();

Parameters default_parameters = GetDefaultParameters();
mThisParameters.ValidateAndAssignDefaults(default_parameters);

KRATOS_CATCH("");
}


~ResidualBasedNewtonRaphsonMPCContactStrategy() override
= default;




typename SolvingStrategyType::Pointer Create(
ModelPart& rModelPart,
Parameters ThisParameters
) const override
{
return Kratos::make_shared<ClassType>(rModelPart, ThisParameters);
}


void Predict() override
{
KRATOS_TRY

BaseType::Predict();

ModelPart& r_model_part = StrategyBaseType::GetModelPart();

TSystemMatrixType& rA = *BaseType::mpA;
TSystemVectorType& rDx = *BaseType::mpDx;
TSystemVectorType& rb = *BaseType::mpb;

TSparseSpace::SetToZero(rA);
TSparseSpace::SetToZero(rDx);
TSparseSpace::SetToZero(rb);

typename TSchemeType::Pointer p_scheme = BaseType::GetScheme();
typename TBuilderAndSolverType::Pointer p_builder_and_solver = BaseType::GetBuilderAndSolver();
p_builder_and_solver->BuildAndSolve(p_scheme, BaseType::GetModelPart(), rA, rDx, rb);

const SizeType echo_level_convergence_criteria = BaseType::mpConvergenceCriteria->GetEchoLevel();
BaseType::mpConvergenceCriteria->SetEchoLevel(0);
mpMPCContactCriteria->PostCriteria(r_model_part, BaseType::GetBuilderAndSolver()->GetDofSet(), rA, rDx, rb);
BaseType::mpConvergenceCriteria->SetEchoLevel(echo_level_convergence_criteria);

KRATOS_CATCH("")
}


void Initialize() override
{
KRATOS_TRY;

ComputeNodalWeights();

BaseType::Initialize();

KRATOS_CATCH("");
}


double Solve() override
{
this->Initialize();
this->InitializeSolutionStep();
this->Predict();
this->SolveSolutionStep();
this->FinalizeSolutionStep(); 

return 0.0;
}


void InitializeSolutionStep() override
{
ComputeNodalWeights();

BaseType::InitializeSolutionStep();

}


void FinalizeSolutionStep() override
{
KRATOS_TRY;

BaseType::FinalizeSolutionStep();

KRATOS_CATCH("");
}


bool SolveSolutionStep() override
{
KRATOS_TRY;

bool is_converged = false;

ModelPart& r_model_part = StrategyBaseType::GetModelPart();

ProcessInfo& r_process_info = r_model_part.GetProcessInfo();

if (r_process_info.Is(INTERACTION)) {
TSystemMatrixType& rA = *BaseType::mpA;
TSystemVectorType& rDx = *BaseType::mpDx;
TSystemVectorType& rb = *BaseType::mpb;

int inner_iteration = 0;
const SizeType echo_level_convergence_criteria = BaseType::mpConvergenceCriteria->GetEchoLevel();
while (!is_converged && inner_iteration < mThisParameters["inner_loop_iterations"].GetInt()) {
++inner_iteration;

if (echo_level_convergence_criteria > 0 && r_model_part.GetCommunicator().MyPID() == 0 ) {
KRATOS_INFO("Simplified semi-smooth strategy") << BOLDFONT("INNER ITERATION: ") << inner_iteration << std::endl;
}

r_process_info[NL_ITERATION_NUMBER] = 1;
is_converged = AuxiliarySolveSolutionStep();

if (r_process_info[NL_ITERATION_NUMBER] == 1) r_process_info[NL_ITERATION_NUMBER] = 2; 
is_converged = mpMPCContactCriteria->PostCriteria(r_model_part, BaseType::GetBuilderAndSolver()->GetDofSet(), rA, rDx, rb);

if (echo_level_convergence_criteria > 0 && r_model_part.GetCommunicator().MyPID() == 0 ) {
if (is_converged) KRATOS_INFO("Simplified semi-smooth strategy") << BOLDFONT("Simplified semi-smooth strategy. INNER ITERATION: ") << BOLDFONT(FGRN("CONVERGED")) << std::endl;
else KRATOS_INFO("Simplified semi-smooth strategy") << BOLDFONT("INNER ITERATION: ") << BOLDFONT(FRED("NOT CONVERGED")) << std::endl;
}
}
} else {
is_converged = AuxiliarySolveSolutionStep();
}

return is_converged;

KRATOS_CATCH("");
}



bool AuxiliarySolveSolutionStep()
{
ModelPart& r_model_part = StrategyBaseType::GetModelPart();
const bool update_each_nl_iteration = mThisParameters["update_each_nl_iteration"].GetBool();
VariableUtils().SetFlag(INTERACTION, update_each_nl_iteration, r_model_part.GetSubModelPart("ComputingContact").Conditions());

typename TSchemeType::Pointer p_scheme = this->GetScheme();
typename TBuilderAndSolverType::Pointer p_builder_and_solver = this->GetBuilderAndSolver();
auto& r_dof_set = p_builder_and_solver->GetDofSet();

TSystemMatrixType& rA  = *BaseType::mpA;
TSystemVectorType& rDx = *BaseType::mpDx;
TSystemVectorType& rb  = *BaseType::mpb;

unsigned int iteration_number = 1;
r_model_part.GetProcessInfo()[NL_ITERATION_NUMBER] = iteration_number;
bool is_converged = false;
bool residual_is_updated = false;

ComputeNodalWeights();

p_scheme->InitializeNonLinIteration(r_model_part, rA, rDx, rb);
BaseType::mpConvergenceCriteria->InitializeNonLinearIteration(r_model_part, r_dof_set, rA, rDx, rb);
is_converged = BaseType::mpConvergenceCriteria->PreCriteria(r_model_part, p_builder_and_solver->GetDofSet(), rA, rDx, rb);


if (StrategyBaseType::mRebuildLevel > 0 || StrategyBaseType::mStiffnessMatrixIsBuilt == false) {
TSparseSpace::SetToZero(rA);
TSparseSpace::SetToZero(rDx);
TSparseSpace::SetToZero(rb);

p_builder_and_solver->BuildAndSolve(p_scheme, r_model_part, rA, rDx, rb);
} else {
TSparseSpace::SetToZero(rDx); 
TSparseSpace::SetToZero(rb);

p_builder_and_solver->BuildRHSAndSolve(p_scheme, r_model_part, rA, rDx, rb);
}

BaseType::EchoInfo(iteration_number);

BaseType::UpdateDatabase(rA, rDx, rb, StrategyBaseType::MoveMeshFlag());

p_scheme->FinalizeNonLinIteration(r_model_part, rA, rDx, rb);
BaseType::mpConvergenceCriteria->FinalizeNonLinearIteration(r_model_part, r_dof_set, rA, rDx, rb);

if (BaseType::mCalculateReactionsFlag)
p_builder_and_solver->CalculateReactions(p_scheme, r_model_part, rA, rDx, rb);

if (is_converged) {
if (BaseType::mpConvergenceCriteria->GetActualizeRHSflag()) {
TSparseSpace::SetToZero(rb);

p_builder_and_solver->BuildRHS(p_scheme, r_model_part, rb);
}

is_converged = BaseType::mpConvergenceCriteria->PostCriteria(r_model_part, p_builder_and_solver->GetDofSet(), rA, rDx, rb);
}

while (!is_converged && iteration_number++ < BaseType::mMaxIterationNumber) {
r_model_part.GetProcessInfo()[NL_ITERATION_NUMBER] = iteration_number;

ComputeNodalWeights();

p_scheme->InitializeNonLinIteration(r_model_part, rA, rDx, rb);
BaseType::mpConvergenceCriteria->InitializeNonLinearIteration(r_model_part, r_dof_set, rA, rDx, rb);

if (update_each_nl_iteration) {
p_builder_and_solver->SetUpDofSet(p_scheme, r_model_part);
p_builder_and_solver->SetUpSystem(r_model_part);
p_builder_and_solver->ResizeAndInitializeVectors(p_scheme, BaseType::mpA, BaseType::mpDx, BaseType::mpb, r_model_part);
}

is_converged = BaseType::mpConvergenceCriteria->PreCriteria(r_model_part, p_builder_and_solver->GetDofSet(), rA, rDx, rb);

if (SparseSpaceType::Size(rDx) != 0) {
if (StrategyBaseType::mRebuildLevel > 1 || !StrategyBaseType::mStiffnessMatrixIsBuilt) {
if (!BaseType::GetKeepSystemConstantDuringIterations()) {
TSparseSpace::SetToZero(rA);
TSparseSpace::SetToZero(rDx);
TSparseSpace::SetToZero(rb);

p_builder_and_solver->BuildAndSolve(p_scheme, r_model_part, rA, rDx, rb);
} else {
TSparseSpace::SetToZero(rDx);
TSparseSpace::SetToZero(rb);

p_builder_and_solver->BuildRHSAndSolve(p_scheme, r_model_part, rA, rDx, rb);
}
} else {
TSparseSpace::SetToZero(rDx);
TSparseSpace::SetToZero(rb);

p_builder_and_solver->BuildRHSAndSolve(p_scheme, r_model_part, rA, rDx, rb);
}
} else {
KRATOS_WARNING("NO DOFS") << "ATTENTION: no free DOFs!! " << std::endl;
}

BaseType::EchoInfo(iteration_number);

BaseType::UpdateDatabase(rA, rDx, rb, StrategyBaseType::MoveMeshFlag());

p_scheme->FinalizeNonLinIteration(r_model_part, rA, rDx, rb);
BaseType::mpConvergenceCriteria->FinalizeNonLinearIteration(r_model_part, r_dof_set, rA, rDx, rb);

residual_is_updated = false;

if (BaseType::mCalculateReactionsFlag)
p_builder_and_solver->CalculateReactions(p_scheme, r_model_part, rA, rDx, rb);

if (is_converged) {
if (BaseType::mpConvergenceCriteria->GetActualizeRHSflag()) {
TSparseSpace::SetToZero(rb);

p_builder_and_solver->BuildRHS(p_scheme, r_model_part, rb);
residual_is_updated = true;
}

is_converged = BaseType::mpConvergenceCriteria->PostCriteria(r_model_part, p_builder_and_solver->GetDofSet(), rA, rDx, rb);
}
}

if (iteration_number >= BaseType::mMaxIterationNumber) {
BaseType::MaxIterationsExceeded();
} else {
KRATOS_INFO_IF("NR-Strategy", this->GetEchoLevel() > 0)  << "Convergence achieved after " << iteration_number << " / "  << BaseType::mMaxIterationNumber << " iterations" << std::endl;
}

if (!residual_is_updated) {

}

if (BaseType::mCalculateReactionsFlag)
p_builder_and_solver->CalculateReactions(p_scheme, r_model_part, rA, rDx, rb);

return is_converged;
}


Parameters GetDefaultParameters() const override
{
Parameters default_parameters = Parameters(R"(
{
"name"                                : "newton_raphson_mpc_contact_strategy",
"inner_loop_iterations"               : 5,
"update_each_nl_iteration"            : false,
"enforce_ntn"                         : false
})" );

const Parameters base_default_parameters = BaseType::GetDefaultParameters();
default_parameters.RecursivelyAddMissingParameters(base_default_parameters);
return default_parameters;
}


static std::string Name()
{
return "newton_raphson_mpc_contact_strategy";
}





protected:



Parameters mThisParameters;                                      
typename TConvergenceCriteriaType::Pointer mpMPCContactCriteria; 




void AssignSettings(const Parameters ThisParameters) override
{
BaseType::AssignSettings(ThisParameters);

mpMPCContactCriteria = Kratos::make_shared<TMPCContactCriteriaType>();

mThisParameters = ThisParameters;
}






ResidualBasedNewtonRaphsonMPCContactStrategy(const ResidualBasedNewtonRaphsonMPCContactStrategy& Other)
{
};

private:






void ComputeNodalWeights()
{
ModelPart& r_contact_model_part = StrategyBaseType::GetModelPart().GetSubModelPart("Contact");

auto& r_nodes_array = r_contact_model_part.Nodes();
VariableUtils().SetNonHistoricalVariableToZero(NODAL_PAUX, r_nodes_array);
VariableUtils().SetNonHistoricalVariableToZero(NODAL_MAUX, r_nodes_array);

auto& r_conditions_array = r_contact_model_part.Conditions();

const bool enforce_ntn = false;

block_for_each(r_conditions_array, [&](Condition& rCond) {
if (rCond.Is(SLAVE)) {
auto& r_geometry = rCond.GetGeometry();
Vector lumping_factor;
lumping_factor = r_geometry.LumpingFactors(lumping_factor);
const double domain_size = r_geometry.DomainSize();
for (IndexType i_node = 0; i_node < r_geometry.size(); ++i_node) {
auto& r_node = r_geometry[i_node];
if (!enforce_ntn) {
AtomicAdd(r_node.GetValue(NODAL_PAUX), 1.0);
}
AtomicAdd(r_node.GetValue(NODAL_MAUX), lumping_factor[i_node] * domain_size);
}
}
});
}




}; 
}  
