
#pragma once


#include <unordered_set>


#ifdef KRATOS_SMP_OPENMP
#include <omp.h>
#endif


#include "includes/define.h"
#include "solving_strategies/builder_and_solvers/residualbased_block_builder_and_solver.h"
#include "includes/model_part.h"
#include "includes/key_hash.h"
#include "utilities/timer.h"
#include "utilities/variable_utils.h"
#include "includes/kratos_flags.h"
#include "includes/lock_object.h"
#include "utilities/sparse_matrix_multiplication_utility.h"
#include "utilities/builtin_timer.h"
#include "utilities/atomic_utilities.h"
#include "spaces/ublas_space.h"

namespace Kratos
{







template<class TSparseSpace,
class TDenseSpace, 
class TLinearSolver
>
class ResidualBasedBlockBuilderAndSolverWithMassAndDamping
: public ResidualBasedBlockBuilderAndSolver< TSparseSpace, TDenseSpace, TLinearSolver >
{
public:

KRATOS_DEFINE_LOCAL_FLAG( SILENT_WARNINGS );

KRATOS_CLASS_POINTER_DEFINITION(ResidualBasedBlockBuilderAndSolverWithMassAndDamping);

using BaseType = ResidualBasedBlockBuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver>;

using TSchemeType = typename BaseType::TSchemeType;
using TSystemMatrixType = typename BaseType::TSystemMatrixType;
using TSystemVectorType = typename BaseType::TSystemVectorType;
using LocalSystemVectorType = typename BaseType::LocalSystemVectorType;
using LocalSystemMatrixType = typename BaseType::LocalSystemMatrixType;
using NodesArrayType = typename BaseType::NodesArrayType;
using ElementsArrayType = typename BaseType::ElementsArrayType;
using ConditionsArrayType = typename BaseType::ConditionsArrayType;

using ElementsContainerType = PointerVectorSet < Element, IndexedObject>;
using EquationIdVectorType = Element::EquationIdVectorType;




ResidualBasedBlockBuilderAndSolverWithMassAndDamping() = default;


explicit ResidualBasedBlockBuilderAndSolverWithMassAndDamping(
typename TLinearSolver::Pointer pNewLinearSystemSolver,
Parameters ThisParameters
) : BaseType(pNewLinearSystemSolver)
{
ThisParameters = this->ValidateAndAssignParameters(ThisParameters, this->GetDefaultParameters());
this->AssignSettings(ThisParameters);
}


explicit ResidualBasedBlockBuilderAndSolverWithMassAndDamping(typename TLinearSolver::Pointer pNewLinearSystemSolver)
: BaseType(pNewLinearSystemSolver)
{
}


~ResidualBasedBlockBuilderAndSolverWithMassAndDamping() override = default;



void Build(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemMatrixType& rA,
TSystemVectorType& rb) override
{
KRATOS_TRY

KRATOS_ERROR_IF(!pScheme) << "No scheme provided!" << std::endl;

const auto nelements = static_cast<int>(rModelPart.Elements().size());

const auto nconditions = static_cast<int>(rModelPart.Conditions().size());

const ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();
ModelPart::ElementsContainerType::iterator el_begin = rModelPart.ElementsBegin();
ModelPart::ConditionsContainerType::iterator cond_begin = rModelPart.ConditionsBegin();

LocalSystemMatrixType lhs_contribution(0, 0);
LocalSystemMatrixType mass_contribution(0, 0);
LocalSystemMatrixType damping_contribution(0, 0);
LocalSystemVectorType rhs_contribution(0);

InitializeDynamicMatrix(mMassMatrix, BaseType::mEquationSystemSize, pScheme, rModelPart);
InitializeDynamicMatrix(mDampingMatrix, BaseType::mEquationSystemSize, pScheme, rModelPart);


Element::EquationIdVectorType equation_ids;

const auto timer = BuiltinTimer();

#pragma omp parallel firstprivate(nelements,nconditions, lhs_contribution, mass_contribution,damping_contribution, rhs_contribution, equation_ids )
{
# pragma omp for  schedule(guided, 512) nowait
for (int k = 0; k < nelements; k++) {
auto it_elem = el_begin + k;

if (it_elem->IsActive()) {
pScheme->CalculateSystemContributions(*it_elem, lhs_contribution, rhs_contribution, equation_ids, r_current_process_info);

it_elem->CalculateMassMatrix(mass_contribution, r_current_process_info);
it_elem->CalculateDampingMatrix(damping_contribution, r_current_process_info);

if (mass_contribution.size1() != 0)
{
BaseType::AssembleLHS(mMassMatrix, mass_contribution, equation_ids);
}
if (damping_contribution.size1() != 0)
{
BaseType::AssembleLHS(mDampingMatrix, damping_contribution, equation_ids);
}


BaseType::Assemble(rA, rb, lhs_contribution, rhs_contribution, equation_ids);
}

}

#pragma omp for  schedule(guided, 512)
for (int k = 0; k < nconditions; k++) {
auto it_cond = cond_begin + k;

if (it_cond->IsActive()) {
pScheme->CalculateSystemContributions(*it_cond, lhs_contribution, rhs_contribution, equation_ids, r_current_process_info);

it_cond->CalculateMassMatrix(mass_contribution, r_current_process_info);
it_cond->CalculateDampingMatrix(damping_contribution, r_current_process_info);

if (mass_contribution.size1() != 0)
{
BaseType::AssembleLHS(mMassMatrix, mass_contribution, equation_ids);
}
if (damping_contribution.size1() != 0)
{
BaseType::AssembleLHS(mDampingMatrix, damping_contribution, equation_ids);
}

BaseType::Assemble(rA, rb, lhs_contribution, rhs_contribution, equation_ids);
}
}
}

KRATOS_INFO_IF("ResidualBasedBlockBuilderAndSolverWithMassAndDamping", this->GetEchoLevel() >= 1) << "Build time: " << timer.ElapsedSeconds() << std::endl;

KRATOS_INFO_IF("ResidualBasedBlockBuilderAndSolverWithMassAndDamping", (this->GetEchoLevel() > 2 && rModelPart.GetCommunicator().MyPID() == 0)) << "Finished parallel building" << std::endl;

KRATOS_CATCH("")
}



void BuildAndSolve(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemMatrixType& rA,
TSystemVectorType& rDx,
TSystemVectorType& rb) override
{
KRATOS_TRY

Timer::Start("Build");

Build(pScheme, rModelPart, rA, rb);

Timer::Stop("Build");

if(rModelPart.MasterSlaveConstraints().size() != 0) {
const auto timer_constraints = BuiltinTimer();
Timer::Start("ApplyConstraints");
BaseType::ApplyConstraints(pScheme, rModelPart, rA, rb);
BaseType::ApplyConstraints(pScheme, rModelPart, mMassMatrix, rb);
BaseType::ApplyConstraints(pScheme, rModelPart, mDampingMatrix, rb);
Timer::Stop("ApplyConstraints");
KRATOS_INFO_IF("ResidualBasedBlockBuilderAndSolverWithMassAndDamping", this->GetEchoLevel() >=1) << "Constraints build time: " << timer_constraints.ElapsedSeconds() << std::endl;
}

BaseType::ApplyDirichletConditions(pScheme, rModelPart, rA, rDx, rb);
BaseType::ApplyDirichletConditions(pScheme, rModelPart, mMassMatrix, rDx, rb);
BaseType::ApplyDirichletConditions(pScheme, rModelPart, mDampingMatrix, rDx, rb);

KRATOS_INFO_IF("ResidualBasedBlockBuilderAndSolverWithMassAndDamping", ( this->GetEchoLevel() == 3)) << "Before the solution of the system" << "\nSystem Matrix = " << rA << "\nUnknowns vector = " << rDx << "\nRHS vector = " << rb << std::endl;

const auto timer = BuiltinTimer();
Timer::Start("Solve");

BaseType::SystemSolveWithPhysics(rA, rDx, rb, rModelPart);

Timer::Stop("Solve");
KRATOS_INFO_IF("ResidualBasedBlockBuilderAndSolverWithMassAndDamping", this->GetEchoLevel() >=1) << "System solve time: " << timer.ElapsedSeconds() << std::endl;

KRATOS_INFO_IF("ResidualBasedBlockBuilderAndSolverWithMassAndDamping", ( this->GetEchoLevel() == 3)) << "After the solution of the system" << "\nSystem Matrix = " << rA << "\nUnknowns vector = " << rDx << "\nRHS vector = " << rb << std::endl;

KRATOS_CATCH("")
}



void BuildRHSAndSolve(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemMatrixType& rA,
TSystemVectorType& rDx,
TSystemVectorType& rb
) override
{
KRATOS_TRY

BuildRHS(pScheme, rModelPart, rb);

if (rModelPart.MasterSlaveConstraints().size() != 0) {
Timer::Start("ApplyRHSConstraints");
BaseType::ApplyRHSConstraints(pScheme, rModelPart, rb);
Timer::Stop("ApplyRHSConstraints");
}

BaseType::ApplyDirichletConditions(pScheme, rModelPart, rA, rDx, rb);

KRATOS_INFO_IF("ResidualBasedBlockBuilderAndSolverWithMassAndDamping", (this->GetEchoLevel() == 3)) << "Before the solution of the system" << "\nSystem Matrix = " << rA << "\nUnknowns vector = " << rDx << "\nRHS vector = " << rb << std::endl;

const auto timer = BuiltinTimer();
Timer::Start("Solve");

BaseType::SystemSolveWithPhysics(rA, rDx, rb, rModelPart);

Timer::Stop("Solve");
KRATOS_INFO_IF("ResidualBasedBlockBuilderAndSolverWithMassAndDamping", this->GetEchoLevel() >= 1) << "System solve time: " << timer.ElapsedSeconds() << std::endl;

KRATOS_INFO_IF("ResidualBasedBlockBuilderAndSolverWithMassAndDamping", (this->GetEchoLevel() == 3)) << "After the solution of the system" << "\nSystem Matrix = " << rA << "\nUnknowns vector = " << rDx << "\nRHS vector = " << rb << std::endl;

KRATOS_CATCH("")
}


void BuildRHS(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemVectorType& rb) override
{
KRATOS_TRY

Timer::Start("BuildRHS");

BuildRHSNoDirichlet(pScheme, rModelPart, rb);

block_for_each(BaseType::mDofSet, [&](Dof<double>& r_dof) {

if (r_dof.IsFixed())
{
const std::size_t i = r_dof.EquationId();
rb[i] = 0.0;
}
});

Timer::Stop("BuildRHS");

KRATOS_CATCH("")
}



Parameters GetDefaultParameters() const override
{
Parameters default_parameters = Parameters(R"(
{
"name"                                 : "block_builder_and_solver_with_mass_and_damping",
"block_builder"                        : true,
"diagonal_values_for_dirichlet_dofs"   : "use_max_diagonal",
"silent_warnings"                      : false
})");

const Parameters base_default_parameters = BaseType::GetDefaultParameters();
default_parameters.RecursivelyAddMissingParameters(base_default_parameters);
return default_parameters;
}


static std::string Name()
{
return "block_builder_and_solver_with_mass_and_damping";
}






std::string Info() const override
{
return "ResidualBasedBlockBuilderAndSolverWithMassAndDamping";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << Info();
}

void PrintData(std::ostream& rOStream) const override
{
rOStream << Info();
}



protected:





void GetFirstAndSecondDerivativeVector(TSystemVectorType& rFirstDerivativeVector, TSystemVectorType& rSecondDerivativeVector, ModelPart& rModelPart)
{
NodesArrayType& r_nodes = rModelPart.Nodes();
const auto n_nodes = static_cast<int>(r_nodes.size());

if (rFirstDerivativeVector.size() != BaseType::mEquationSystemSize) {
rFirstDerivativeVector.resize(BaseType::mEquationSystemSize, false);
}
TSparseSpace::SetToZero(rFirstDerivativeVector);

if (rSecondDerivativeVector.size() != BaseType::mEquationSystemSize) {
rSecondDerivativeVector.resize(BaseType::mEquationSystemSize, false);
}
TSparseSpace::SetToZero(rSecondDerivativeVector);

std::size_t id_x;
std::size_t id_y;
std::size_t id_z;

#pragma omp parallel firstprivate(n_nodes) private(id_x, id_y, id_z)
{
#pragma omp for schedule(guided, 512) nowait
for (int i = 0; i < n_nodes; i++) {
typename NodesArrayType::iterator it = r_nodes.begin() + i;
if (it->IsActive()) {

id_x = it->GetDof(DISPLACEMENT_X).EquationId();
id_y = it->GetDof(DISPLACEMENT_Y).EquationId();

rFirstDerivativeVector[id_x] = it->FastGetSolutionStepValue(VELOCITY_X);
rFirstDerivativeVector[id_y] = it->FastGetSolutionStepValue(VELOCITY_Y);

rSecondDerivativeVector[id_x] = it->FastGetSolutionStepValue(ACCELERATION_X);
rSecondDerivativeVector[id_y] = it->FastGetSolutionStepValue(ACCELERATION_Y);

if (it->HasDofFor(DISPLACEMENT_Z))
{
id_z = it->GetDof(DISPLACEMENT_Z).EquationId();

rFirstDerivativeVector[id_z] = it->FastGetSolutionStepValue(VELOCITY_Z);
rSecondDerivativeVector[id_z] = it->FastGetSolutionStepValue(ACCELERATION_Z);
}
}
}
}
}


void BuildRHSNoDirichlet(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemVectorType& rb)
{
KRATOS_TRY

ElementsArrayType& r_elements = rModelPart.Elements();

ConditionsArrayType& r_conditions = rModelPart.Conditions();

const ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();

LocalSystemVectorType rhs_contribution = LocalSystemVectorType(0);

Element::EquationIdVectorType equation_ids;


const int nelements = static_cast<int>(r_elements.size());
#pragma omp parallel firstprivate(nelements, rhs_contribution, equation_ids)
{
#pragma omp for schedule(guided, 512) nowait
for (int i=0; i<nelements; i++) {
typename ElementsArrayType::iterator it = r_elements.begin() + i;
if(it->IsActive()) {

it->CalculateRightHandSide(rhs_contribution, r_current_process_info);
it->EquationIdVector(equation_ids, r_current_process_info);

BaseType::AssembleRHS(rb, rhs_contribution, equation_ids);
}
}

rhs_contribution.resize(0, false);

const int nconditions = static_cast<int>(r_conditions.size());
#pragma omp for schedule(guided, 512)
for (int i = 0; i<nconditions; i++) {
auto it = r_conditions.begin() + i;
if(it->IsActive()) {

it->CalculateRightHandSide(rhs_contribution, r_current_process_info);
it->EquationIdVector(equation_ids, r_current_process_info);

BaseType::AssembleRHS(rb, rhs_contribution, equation_ids);

}
}
}

AddMassAndDampingToRhs(rModelPart, rb);

KRATOS_CATCH("")

}


private:

TSystemMatrixType mMassMatrix;
TSystemMatrixType mDampingMatrix;

void InitializeDynamicMatrix(TSystemMatrixType& rMatrix, unsigned int MatrixSize, typename TSchemeType::Pointer pScheme, ModelPart& rModelPart)
{
rMatrix.resize(MatrixSize, MatrixSize, false);
BaseType::ConstructMatrixStructure(pScheme, rMatrix, rModelPart);
TSparseSpace::SetToZero(rMatrix);
}

void CalculateAndAddDynamicContributionToRhs(TSystemVectorType& rSolutionVector,TSystemMatrixType& rGlobalMatrix, TSystemVectorType& rb)
{
TSystemVectorType contribution;
contribution.resize(BaseType::mEquationSystemSize, false);
TSparseSpace::SetToZero(contribution);
TSparseSpace::Mult(rGlobalMatrix, rSolutionVector, contribution);

TSparseSpace::UnaliasedAdd(rb, -1.0, contribution);
}


void AddMassAndDampingToRhs(ModelPart& rModelPart, TSystemVectorType& rb)
{

TSystemVectorType first_derivative_vector;
TSystemVectorType second_derivative_vector;
GetFirstAndSecondDerivativeVector(first_derivative_vector, second_derivative_vector, rModelPart);

CalculateAndAddDynamicContributionToRhs(second_derivative_vector, mMassMatrix, rb);
CalculateAndAddDynamicContributionToRhs(first_derivative_vector, mDampingMatrix, rb);

}


}; 



template<class TSparseSpace, class TDenseSpace, class TLinearSolver>
const Kratos::Flags ResidualBasedBlockBuilderAndSolverWithMassAndDamping<TSparseSpace, TDenseSpace, TLinearSolver>::SILENT_WARNINGS(Kratos::Flags::Create(0));


} 
