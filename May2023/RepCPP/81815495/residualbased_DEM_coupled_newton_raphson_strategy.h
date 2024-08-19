
#if !defined(KRATOS_RESIDUALBASED_DEM_COUPLED_NEWTON_RAPHSON_STRATEGY)
#define KRATOS_RESIDUALBASED_DEM_COUPLED_NEWTON_RAPHSON_STRATEGY



#include "includes/define.h"
#include "solving_strategies/strategies/residualbased_newton_raphson_strategy.h"
#include "solving_strategies/convergencecriterias/convergence_criteria.h"
#include "utilities/builtin_timer.h"
#include "custom_strategies/strategies/explicit_solver_strategy.h"

#include "custom_processes/update_dem_kinematics_process.h"
#include "custom_processes/transfer_nodal_forces_to_fem.h"

#include "solving_strategies/builder_and_solvers/residualbased_block_builder_and_solver.h"

namespace Kratos
{








template <class TSparseSpace,
class TDenseSpace,  
class TLinearSolver 
>
class ResidualBasedDEMCoupledNewtonRaphsonStrategy
: public ResidualBasedNewtonRaphsonStrategy<TSparseSpace, TDenseSpace, TLinearSolver>
{
public:
typedef ConvergenceCriteria<TSparseSpace, TDenseSpace> TConvergenceCriteriaType;

typedef ExplicitSolverStrategy ExplicitSolverStrategyType;

KRATOS_CLASS_POINTER_DEFINITION(ResidualBasedDEMCoupledNewtonRaphsonStrategy);

typedef ResidualBasedNewtonRaphsonStrategy<TSparseSpace, TDenseSpace, TLinearSolver> BaseType;
typedef ResidualBasedDEMCoupledNewtonRaphsonStrategy<TSparseSpace, TDenseSpace, TLinearSolver> ClassType;
typedef typename BaseType::TBuilderAndSolverType TBuilderAndSolverType;
typedef typename BaseType::TDataType TDataType;
typedef TSparseSpace SparseSpaceType;
typedef typename BaseType::TSchemeType TSchemeType;
typedef typename BaseType::DofsArrayType DofsArrayType;
typedef typename BaseType::TSystemMatrixType TSystemMatrixType;
typedef typename BaseType::TSystemVectorType TSystemVectorType;
typedef typename BaseType::LocalSystemVectorType LocalSystemVectorType;
typedef typename BaseType::LocalSystemMatrixType LocalSystemMatrixType;
typedef typename BaseType::TSystemMatrixPointerType TSystemMatrixPointerType;
typedef typename BaseType::TSystemVectorPointerType TSystemVectorPointerType;



explicit ResidualBasedDEMCoupledNewtonRaphsonStrategy() : BaseType()
{
}


explicit ResidualBasedDEMCoupledNewtonRaphsonStrategy(
ModelPart& rModelPart,
ExplicitSolverStrategyType::Pointer pDEMStrategy,
typename TSchemeType::Pointer pScheme,
typename TConvergenceCriteriaType::Pointer pNewConvergenceCriteria,
typename TBuilderAndSolverType::Pointer pNewBuilderAndSolver,
int MaxIterations = 30,
bool CalculateReactions = false,
bool ReformDofSetAtEachStep = false,
bool MoveMeshFlag = false)
: BaseType(rModelPart,
pScheme,
pNewConvergenceCriteria,
pNewBuilderAndSolver,
MaxIterations,
CalculateReactions,
ReformDofSetAtEachStep,
MoveMeshFlag),
mpDEMStrategy(pDEMStrategy)
{
}



~ResidualBasedDEMCoupledNewtonRaphsonStrategy() override
{
auto p_builder_and_solver = this->GetBuilderAndSolver();
if (p_builder_and_solver != nullptr) {
p_builder_and_solver->Clear();
}

this->mpA.reset();
this->mpDx.reset();
this->mpb.reset();

this->Clear();
}

/



void Initialize() override
{
KRATOS_TRY;
BaseType::Initialize();
mpDEMStrategy->Initialize();
KRATOS_CATCH("");
}


void InitializeSolutionStep() override
{
KRATOS_TRY;
BaseType::InitializeSolutionStep();
mpDEMStrategy->InitializeSolutionStep();
TransferNodalForcesToFem(this->GetModelPart(), false).Execute();
UpdateDemKinematicsProcess(this->GetModelPart()).Execute();
KRATOS_CATCH("");
}


void FinalizeSolutionStep() override
{
KRATOS_TRY;
BaseType::FinalizeSolutionStep();
mpDEMStrategy->FinalizeSolutionStep();
KRATOS_CATCH("");
}


bool SolveSolutionStep() override
{
auto update_dem_kinematics_process = UpdateDemKinematicsProcess(this->GetModelPart());
update_dem_kinematics_process.Execute();
mpDEMStrategy->SolveSolutionStep();
auto transfer_process = TransferNodalForcesToFem(this->GetModelPart(), false);
transfer_process.Execute();
update_dem_kinematics_process.Execute();

ModelPart& r_model_part = BaseType::GetModelPart();
typename TSchemeType::Pointer p_scheme = this->GetScheme();
typename TBuilderAndSolverType::Pointer p_builder_and_solver = this->GetBuilderAndSolver();
auto& r_dof_set = p_builder_and_solver->GetDofSet();

TSystemMatrixType& rA  = *this->mpA;
TSystemVectorType& rDx = *this->mpDx;
TSystemVectorType& rb  = *this->mpb;

unsigned int iteration_number = 1;
r_model_part.GetProcessInfo()[NL_ITERATION_NUMBER] = iteration_number;
bool residual_is_updated = false;
p_scheme->InitializeNonLinIteration(r_model_part, rA, rDx, rb);
this->mpConvergenceCriteria->InitializeNonLinearIteration(r_model_part, r_dof_set, rA, rDx, rb);
bool is_converged = this->mpConvergenceCriteria->PreCriteria(r_model_part, r_dof_set, rA, rDx, rb);

if (BaseType::mRebuildLevel > 0 || BaseType::mStiffnessMatrixIsBuilt == false) {
TSparseSpace::SetToZero(rA);
TSparseSpace::SetToZero(rDx);
TSparseSpace::SetToZero(rb);

if (this->mUseOldStiffnessInFirstIteration){
p_builder_and_solver->BuildAndSolveLinearizedOnPreviousIteration(p_scheme, r_model_part, rA, rDx, rb, BaseType::MoveMeshFlag());
} else {
p_builder_and_solver->BuildAndSolve(p_scheme, r_model_part, rA, rDx, rb);
}
} else {
TSparseSpace::SetToZero(rDx);  
TSparseSpace::SetToZero(rb);

p_builder_and_solver->BuildRHSAndSolve(p_scheme, r_model_part, rA, rDx, rb);
}

this->EchoInfo(iteration_number);

this->UpdateDatabase(rA, rDx, rb, BaseType::MoveMeshFlag());

p_scheme->FinalizeNonLinIteration(r_model_part, rA, rDx, rb);
this->mpConvergenceCriteria->FinalizeNonLinearIteration(r_model_part, r_dof_set, rA, rDx, rb);

if (is_converged) {
if (this->mpConvergenceCriteria->GetActualizeRHSflag()) {
TSparseSpace::SetToZero(rb);
p_builder_and_solver->BuildRHS(p_scheme, r_model_part, rb);
}
is_converged = this->mpConvergenceCriteria->PostCriteria(r_model_part, r_dof_set, rA, rDx, rb);
}

while (is_converged == false && iteration_number++ < this->mMaxIterationNumber) {
update_dem_kinematics_process.Execute();
mpDEMStrategy->SolveSolutionStep();
auto transfer_process_damped = TransferNodalForcesToFem(this->GetModelPart(), true);
transfer_process_damped.Execute();
update_dem_kinematics_process.Execute();

r_model_part.GetProcessInfo()[NL_ITERATION_NUMBER] = iteration_number;

p_scheme->InitializeNonLinIteration(r_model_part, rA, rDx, rb);
this->mpConvergenceCriteria->InitializeNonLinearIteration(r_model_part, r_dof_set, rA, rDx, rb);

is_converged = this->mpConvergenceCriteria->PreCriteria(r_model_part, r_dof_set, rA, rDx, rb);

if (SparseSpaceType::Size(rDx) != 0) {
if (BaseType::mRebuildLevel > 1 || BaseType::mStiffnessMatrixIsBuilt == false) {
if (this->GetKeepSystemConstantDuringIterations() == false) {
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

this->EchoInfo(iteration_number);

this->UpdateDatabase(rA, rDx, rb, BaseType::MoveMeshFlag());

p_scheme->FinalizeNonLinIteration(r_model_part, rA, rDx, rb);
this->mpConvergenceCriteria->FinalizeNonLinearIteration(r_model_part, r_dof_set, rA, rDx, rb);

residual_is_updated = false;

if (is_converged == true) {
if (this->mpConvergenceCriteria->GetActualizeRHSflag() == true) {
TSparseSpace::SetToZero(rb);
p_builder_and_solver->BuildRHS(p_scheme, r_model_part, rb);
residual_is_updated = true;
}
is_converged = this->mpConvergenceCriteria->PostCriteria(r_model_part, r_dof_set, rA, rDx, rb);
}
}

if (iteration_number >= this->mMaxIterationNumber) {
this->MaxIterationsExceeded();
} else {
KRATOS_INFO_IF("ResidualBasedDEMCoupledNewtonRaphsonStrategy", this->GetEchoLevel() > 0)
<< "Convergence achieved after " << iteration_number << " / "
<< this->mMaxIterationNumber << " iterations" << std::endl;
}

if (this->mCalculateReactionsFlag == true)
p_builder_and_solver->CalculateReactions(p_scheme, r_model_part, rA, rDx, rb);

return is_converged;
}


void Predict() override
{
KRATOS_TRY
const DataCommunicator &r_comm = BaseType::GetModelPart().GetCommunicator().GetDataCommunicator();
if (this->mInitializeWasPerformed == false)
BaseType::Initialize();

BaseType::InitializeSolutionStep();
mpDEMStrategy->InitializeSolutionStep();

TSystemMatrixType& rA  = *this->mpA;
TSystemVectorType& rDx = *this->mpDx;
TSystemVectorType& rb  = *this->mpb;

DofsArrayType& r_dof_set = this->GetBuilderAndSolver()->GetDofSet();

this->GetScheme()->Predict(BaseType::GetModelPart(), r_dof_set, rA, rDx, rb);

auto& r_constraints_array = BaseType::GetModelPart().MasterSlaveConstraints();
const int local_number_of_constraints = r_constraints_array.size();
const int global_number_of_constraints = r_comm.SumAll(local_number_of_constraints);
if (global_number_of_constraints != 0) {
const auto& r_process_info = BaseType::GetModelPart().GetProcessInfo();

const auto it_const_begin = r_constraints_array.begin();

#pragma omp parallel for
for (int i=0; i<static_cast<int>(local_number_of_constraints); ++i)
(it_const_begin + i)->ResetSlaveDofs(r_process_info);

#pragma omp parallel for
for (int i=0; i<static_cast<int>(local_number_of_constraints); ++i)
(it_const_begin + i)->Apply(r_process_info);

TSparseSpace::SetToZero(rDx);
this->GetScheme()->Update(BaseType::GetModelPart(), r_dof_set, rA, rDx, rb);
}

if (this->MoveMeshFlag() == true)
BaseType::MoveMesh();

KRATOS_CATCH("")
}









std::string Info() const override
{
return "ResidualBasedDEMCoupledNewtonRaphsonStrategy";
}




private:








protected:



typename ExplicitSolverStrategy::Pointer mpDEMStrategy = nullptr;








ResidualBasedDEMCoupledNewtonRaphsonStrategy(const ResidualBasedDEMCoupledNewtonRaphsonStrategy &Other){};


}; 




} 

#endif 
