
#pragma once


#include <unordered_set>


#ifdef KRATOS_SMP_OPENMP
#include <omp.h>
#endif


#include "includes/define.h"
#include "solving_strategies/builder_and_solvers/builder_and_solver.h"
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
class ResidualBasedBlockBuilderAndSolver
: public BuilderAndSolver< TSparseSpace, TDenseSpace, TLinearSolver >
{
public:

KRATOS_DEFINE_LOCAL_FLAG( SILENT_WARNINGS );

KRATOS_CLASS_POINTER_DEFINITION(ResidualBasedBlockBuilderAndSolver);

typedef BuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver> BaseType;

typedef ResidualBasedBlockBuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver> ClassType;

typedef std::size_t SizeType;
typedef std::size_t IndexType;

typedef typename BaseType::TSchemeType TSchemeType;
typedef typename BaseType::TDataType TDataType;
typedef typename BaseType::DofsArrayType DofsArrayType;
typedef typename BaseType::TSystemMatrixType TSystemMatrixType;
typedef typename BaseType::TSystemVectorType TSystemVectorType;
typedef typename BaseType::LocalSystemVectorType LocalSystemVectorType;
typedef typename BaseType::LocalSystemMatrixType LocalSystemMatrixType;
typedef typename BaseType::TSystemMatrixPointerType TSystemMatrixPointerType;
typedef typename BaseType::TSystemVectorPointerType TSystemVectorPointerType;
typedef typename BaseType::NodesArrayType NodesArrayType;
typedef typename BaseType::ElementsArrayType ElementsArrayType;
typedef typename BaseType::ConditionsArrayType ConditionsArrayType;

typedef PointerVectorSet<Element, IndexedObject> ElementsContainerType;
typedef Element::EquationIdVectorType EquationIdVectorType;
typedef Element::DofsVectorType DofsVectorType;
typedef boost::numeric::ublas::compressed_matrix<double> CompressedMatrixType;

typedef Node NodeType;
typedef typename NodeType::DofType DofType;
typedef typename DofType::Pointer DofPointerType;



explicit ResidualBasedBlockBuilderAndSolver() : BaseType()
{
}


explicit ResidualBasedBlockBuilderAndSolver(
typename TLinearSolver::Pointer pNewLinearSystemSolver,
Parameters ThisParameters
) : BaseType(pNewLinearSystemSolver)
{
ThisParameters = this->ValidateAndAssignParameters(ThisParameters, this->GetDefaultParameters());
this->AssignSettings(ThisParameters);
}


explicit ResidualBasedBlockBuilderAndSolver(typename TLinearSolver::Pointer pNewLinearSystemSolver)
: BaseType(pNewLinearSystemSolver)
{
}


~ResidualBasedBlockBuilderAndSolver() override
{
}


typename BaseType::Pointer Create(
typename TLinearSolver::Pointer pNewLinearSystemSolver,
Parameters ThisParameters
) const override
{
return Kratos::make_shared<ClassType>(pNewLinearSystemSolver,ThisParameters);
}




void Build(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemMatrixType& A,
TSystemVectorType& b) override
{
KRATOS_TRY

KRATOS_ERROR_IF(!pScheme) << "No scheme provided!" << std::endl;

const int nelements = static_cast<int>(rModelPart.Elements().size());

const int nconditions = static_cast<int>(rModelPart.Conditions().size());

const ProcessInfo& CurrentProcessInfo = rModelPart.GetProcessInfo();
ModelPart::ElementsContainerType::iterator el_begin = rModelPart.ElementsBegin();
ModelPart::ConditionsContainerType::iterator cond_begin = rModelPart.ConditionsBegin();

LocalSystemMatrixType LHS_Contribution = LocalSystemMatrixType(0, 0);
LocalSystemVectorType RHS_Contribution = LocalSystemVectorType(0);

Element::EquationIdVectorType EquationId;

const auto timer = BuiltinTimer();

#pragma omp parallel firstprivate(nelements,nconditions, LHS_Contribution, RHS_Contribution, EquationId )
{
# pragma omp for  schedule(guided, 512) nowait
for (int k = 0; k < nelements; k++) {
auto it_elem = el_begin + k;

if (it_elem->IsActive()) {
pScheme->CalculateSystemContributions(*it_elem, LHS_Contribution, RHS_Contribution, EquationId, CurrentProcessInfo);

Assemble(A, b, LHS_Contribution, RHS_Contribution, EquationId);
}

}

#pragma omp for  schedule(guided, 512)
for (int k = 0; k < nconditions; k++) {
auto it_cond = cond_begin + k;

if (it_cond->IsActive()) {
pScheme->CalculateSystemContributions(*it_cond, LHS_Contribution, RHS_Contribution, EquationId, CurrentProcessInfo);

Assemble(A, b, LHS_Contribution, RHS_Contribution, EquationId);
}
}
}

KRATOS_INFO_IF("ResidualBasedBlockBuilderAndSolver", this->GetEchoLevel() >= 1) << "Build time: " << timer.ElapsedSeconds() << std::endl;

KRATOS_INFO_IF("ResidualBasedBlockBuilderAndSolver", (this->GetEchoLevel() > 2 && rModelPart.GetCommunicator().MyPID() == 0)) << "Finished parallel building" << std::endl;

KRATOS_CATCH("")
}


void BuildLHS(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemMatrixType& rA
) override
{
KRATOS_TRY

KRATOS_ERROR_IF(!pScheme) << "No scheme provided!" << std::endl;

const int nelements = static_cast<int>(rModelPart.Elements().size());

const int nconditions = static_cast<int>(rModelPart.Conditions().size());

const ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();
const auto it_elem_begin = rModelPart.ElementsBegin();
const auto it_cond_begin = rModelPart.ConditionsBegin();

LocalSystemMatrixType lhs_contribution(0, 0);

Element::EquationIdVectorType equation_id;

const auto timer = BuiltinTimer();

#pragma omp parallel firstprivate(nelements, nconditions, lhs_contribution, equation_id )
{
# pragma omp for  schedule(guided, 512) nowait
for (int k = 0; k < nelements; ++k) {
auto it_elem = it_elem_begin + k;

if (it_elem->IsActive()) {
pScheme->CalculateLHSContribution(*it_elem, lhs_contribution, equation_id, r_current_process_info);

AssembleLHS(rA, lhs_contribution, equation_id);
}
}

#pragma omp for  schedule(guided, 512)
for (int k = 0; k < nconditions; ++k) {
auto it_cond = it_cond_begin + k;

if (it_cond->IsActive()) {
pScheme->CalculateLHSContribution(*it_cond, lhs_contribution, equation_id, r_current_process_info);

AssembleLHS(rA, lhs_contribution, equation_id);
}
}
}

KRATOS_INFO_IF("ResidualBasedBlockBuilderAndSolver", this->GetEchoLevel() >= 1) << "Build time LHS: " << timer.ElapsedSeconds() << std::endl;

KRATOS_INFO_IF("ResidualBasedBlockBuilderAndSolver", this->GetEchoLevel() > 2) << "Finished parallel building LHS" << std::endl;

KRATOS_CATCH("")
}


void BuildLHS_CompleteOnFreeRows(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemMatrixType& A) override
{
KRATOS_TRY

TSystemVectorType tmp(A.size1(), 0.0);
this->Build(pScheme, rModelPart, A, tmp);

KRATOS_CATCH("")
}


void SystemSolve(
TSystemMatrixType& A,
TSystemVectorType& Dx,
TSystemVectorType& b
) override
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

if(mT.size1() != 0) 
{
TSystemVectorType Dxmodified = Dx;

TSparseSpace::Mult(mT, Dxmodified, Dx);
}

KRATOS_INFO_IF("ResidualBasedBlockBuilderAndSolver", this->GetEchoLevel() > 1) << *(BaseType::mpLinearSystemSolver) << std::endl;

KRATOS_CATCH("")
}


virtual void SystemSolveWithPhysics(
TSystemMatrixType& rA,
TSystemVectorType& rDx,
TSystemVectorType& rb,
ModelPart& rModelPart
)
{
if(rModelPart.MasterSlaveConstraints().size() != 0) {
TSystemVectorType Dxmodified(rb.size());

TSparseSpace::SetToZero(Dxmodified);

InternalSystemSolveWithPhysics(rA, Dxmodified, rb, rModelPart);

TSparseSpace::Mult(mT, Dxmodified, rDx);
} else {
InternalSystemSolveWithPhysics(rA, rDx, rb, rModelPart);
}
}


void InternalSystemSolveWithPhysics(
TSystemMatrixType& A,
TSystemVectorType& Dx,
TSystemVectorType& b,
ModelPart& rModelPart
)
{
KRATOS_TRY

double norm_b;
if (TSparseSpace::Size(b) != 0)
norm_b = TSparseSpace::TwoNorm(b);
else
norm_b = 0.00;

if (norm_b != 0.00) {
if(BaseType::mpLinearSystemSolver->AdditionalPhysicalDataIsNeeded() )
BaseType::mpLinearSystemSolver->ProvideAdditionalData(A, Dx, b, BaseType::mDofSet, rModelPart);

BaseType::mpLinearSystemSolver->Solve(A, Dx, b);
} else {
KRATOS_WARNING_IF("ResidualBasedBlockBuilderAndSolver", mOptions.IsNot(SILENT_WARNINGS)) << "ATTENTION! setting the RHS to zero!" << std::endl;
}

KRATOS_INFO_IF("ResidualBasedBlockBuilderAndSolver", this->GetEchoLevel() > 1) << *(BaseType::mpLinearSystemSolver) << std::endl;

KRATOS_CATCH("")
}


void BuildAndSolve(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemMatrixType& A,
TSystemVectorType& Dx,
TSystemVectorType& b) override
{
KRATOS_TRY

Timer::Start("Build");

Build(pScheme, rModelPart, A, b);

Timer::Stop("Build");

if(rModelPart.MasterSlaveConstraints().size() != 0) {
const auto timer_constraints = BuiltinTimer();
Timer::Start("ApplyConstraints");
ApplyConstraints(pScheme, rModelPart, A, b);
Timer::Stop("ApplyConstraints");
KRATOS_INFO_IF("ResidualBasedBlockBuilderAndSolver", this->GetEchoLevel() >=1) << "Constraints build time: " << timer_constraints.ElapsedSeconds() << std::endl;
}

ApplyDirichletConditions(pScheme, rModelPart, A, Dx, b);

KRATOS_INFO_IF("ResidualBasedBlockBuilderAndSolver", ( this->GetEchoLevel() == 3)) << "Before the solution of the system" << "\nSystem Matrix = " << A << "\nUnknowns vector = " << Dx << "\nRHS vector = " << b << std::endl;

const auto timer = BuiltinTimer();
Timer::Start("Solve");

SystemSolveWithPhysics(A, Dx, b, rModelPart);

Timer::Stop("Solve");
KRATOS_INFO_IF("ResidualBasedBlockBuilderAndSolver", this->GetEchoLevel() >=1) << "System solve time: " << timer.ElapsedSeconds() << std::endl;

KRATOS_INFO_IF("ResidualBasedBlockBuilderAndSolver", ( this->GetEchoLevel() == 3)) << "After the solution of the system" << "\nSystem Matrix = " << A << "\nUnknowns vector = " << Dx << "\nRHS vector = " << b << std::endl;

KRATOS_CATCH("")
}


void BuildAndSolveLinearizedOnPreviousIteration(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemMatrixType& rA,
TSystemVectorType& rDx,
TSystemVectorType& rb,
const bool MoveMesh
) override
{
Timer::Start("Linearizing on Old iteration");

KRATOS_INFO_IF("BlockBuilderAndSolver", this->GetEchoLevel() > 0) << "Linearizing on Old iteration" << std::endl;

KRATOS_ERROR_IF(rModelPart.GetBufferSize() == 1) << "BlockBuilderAndSolver: \n"
<< "The buffer size needs to be at least 2 in order to use \n"
<< "BuildAndSolveLinearizedOnPreviousIteration \n"
<< "current buffer size for modelpart: " << rModelPart.Name() << std::endl
<< "is :" << rModelPart.GetBufferSize()
<< " Please set IN THE STRATEGY SETTINGS "
<< " UseOldStiffnessInFirstIteration=false " << std::endl;

DofsArrayType fixed_dofs;
for(auto& r_dof : BaseType::mDofSet){
if(r_dof.IsFixed()){
fixed_dofs.push_back(&r_dof);
r_dof.FreeDof();
}
}

TSystemVectorType dx_prediction(rDx);
TSystemVectorType rhs_addition(rb); 

block_for_each(BaseType::mDofSet, [&](Dof<double>& rDof){
dx_prediction[rDof.EquationId()] = -(rDof.GetSolutionStepValue() - rDof.GetSolutionStepValue(1));

});

pScheme->Update(rModelPart, BaseType::mDofSet, rA, dx_prediction, rb);
if (MoveMesh) {
VariableUtils().UpdateCurrentPosition(rModelPart.Nodes(),DISPLACEMENT,0);
}

Timer::Stop("Linearizing on Old iteration");

Timer::Start("Build");

this->Build(pScheme, rModelPart, rA, rb);

Timer::Stop("Build");

TSparseSpace::InplaceMult(dx_prediction, -1.0); 
TSparseSpace::UnaliasedAdd(rDx, 1.0, dx_prediction);

pScheme->Update(rModelPart, BaseType::mDofSet, rA, dx_prediction, rb);
if (MoveMesh) {
VariableUtils().UpdateCurrentPosition(rModelPart.Nodes(),DISPLACEMENT,0);
}

TSparseSpace::Mult(rA, dx_prediction, rhs_addition);
TSparseSpace::UnaliasedAdd(rb, -1.0, rhs_addition);

for(auto& dof : fixed_dofs)
dof.FixDof();

if (!rModelPart.MasterSlaveConstraints().empty()) {
const auto timer_constraints = BuiltinTimer();
Timer::Start("ApplyConstraints");
this->ApplyConstraints(pScheme, rModelPart, rA, rb);
Timer::Stop("ApplyConstraints");
KRATOS_INFO_IF("ResidualBasedBlockBuilderAndSolver", this->GetEchoLevel() >=1) << "Constraints build time: " << timer_constraints.ElapsedSeconds() << std::endl;
}
this->ApplyDirichletConditions(pScheme, rModelPart, rA, rDx, rb);

KRATOS_INFO_IF("ResidualBasedBlockBuilderAndSolver", ( this->GetEchoLevel() == 3)) << "Before the solution of the system" << "\nSystem Matrix = " << rA << "\nUnknowns vector = " << rDx << "\nRHS vector = " << rb << std::endl;

const auto timer = BuiltinTimer();

Timer::Start("Solve");

this->SystemSolveWithPhysics(rA, rDx, rb, rModelPart);

Timer::Stop("Solve");
KRATOS_INFO_IF("ResidualBasedBlockBuilderAndSolver", this->GetEchoLevel() >=1) << "System solve time: " << timer.ElapsedSeconds() << std::endl;

KRATOS_INFO_IF("ResidualBasedBlockBuilderAndSolver", ( this->GetEchoLevel() == 3)) << "After the solution of the system" << "\nSystem Matrix = " << rA << "\nUnknowns vector = " << rDx << "\nRHS vector = " << rb << std::endl;
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

if(rModelPart.MasterSlaveConstraints().size() != 0) {
Timer::Start("ApplyRHSConstraints");
ApplyRHSConstraints(pScheme, rModelPart, rb);
Timer::Stop("ApplyRHSConstraints");
}

ApplyDirichletConditions(pScheme, rModelPart, rA, rDx, rb);

KRATOS_INFO_IF("ResidualBasedBlockBuilderAndSolver", ( this->GetEchoLevel() == 3)) << "Before the solution of the system" << "\nSystem Matrix = " << rA << "\nUnknowns vector = " << rDx << "\nRHS vector = " << rb << std::endl;

const auto timer = BuiltinTimer();
Timer::Start("Solve");

SystemSolveWithPhysics(rA, rDx, rb, rModelPart);

Timer::Stop("Solve");
KRATOS_INFO_IF("ResidualBasedBlockBuilderAndSolver", this->GetEchoLevel() >=1) << "System solve time: " << timer.ElapsedSeconds() << std::endl;

KRATOS_INFO_IF("ResidualBasedBlockBuilderAndSolver", ( this->GetEchoLevel() == 3)) << "After the solution of the system" << "\nSystem Matrix = " << rA << "\nUnknowns vector = " << rDx << "\nRHS vector = " << rb << std::endl;

KRATOS_CATCH("")
}


void BuildRHS(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemVectorType& b) override
{
KRATOS_TRY

Timer::Start("BuildRHS");

BuildRHSNoDirichlet(pScheme,rModelPart,b);

block_for_each(BaseType::mDofSet, [&](Dof<double>& rDof){
const std::size_t i = rDof.EquationId();

if (rDof.IsFixed())
b[i] = 0.0;
});

Timer::Stop("BuildRHS");

KRATOS_CATCH("")
}


void SetUpDofSet(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart
) override
{
KRATOS_TRY;

KRATOS_INFO_IF("ResidualBasedBlockBuilderAndSolver", ( this->GetEchoLevel() > 1 && rModelPart.GetCommunicator().MyPID() == 0)) << "Setting up the dofs" << std::endl;

ElementsArrayType& r_elements_array = rModelPart.Elements();
const int number_of_elements = static_cast<int>(r_elements_array.size());

DofsVectorType dof_list, second_dof_list; 

unsigned int nthreads = ParallelUtilities::GetNumThreads();

typedef std::unordered_set < NodeType::DofType::Pointer, DofPointerHasher>  set_type;

KRATOS_INFO_IF("ResidualBasedBlockBuilderAndSolver", ( this->GetEchoLevel() > 2)) << "Number of threads" << nthreads << "\n" << std::endl;

KRATOS_INFO_IF("ResidualBasedBlockBuilderAndSolver", ( this->GetEchoLevel() > 2)) << "Initializing element loop" << std::endl;


set_type dof_global_set;
dof_global_set.reserve(number_of_elements*20);

#pragma omp parallel firstprivate(dof_list, second_dof_list)
{
const ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();

set_type dofs_tmp_set;
dofs_tmp_set.reserve(20000);

#pragma omp for schedule(guided, 512) nowait
for (int i = 0; i < number_of_elements; ++i) {
auto it_elem = r_elements_array.begin() + i;

pScheme->GetDofList(*it_elem, dof_list, r_current_process_info);
dofs_tmp_set.insert(dof_list.begin(), dof_list.end());
}

ConditionsArrayType& r_conditions_array = rModelPart.Conditions();
const int number_of_conditions = static_cast<int>(r_conditions_array.size());
#pragma omp for  schedule(guided, 512) nowait
for (int i = 0; i < number_of_conditions; ++i) {
auto it_cond = r_conditions_array.begin() + i;

pScheme->GetDofList(*it_cond, dof_list, r_current_process_info);
dofs_tmp_set.insert(dof_list.begin(), dof_list.end());
}

auto& r_constraints_array = rModelPart.MasterSlaveConstraints();
const int number_of_constraints = static_cast<int>(r_constraints_array.size());
#pragma omp for  schedule(guided, 512) nowait
for (int i = 0; i < number_of_constraints; ++i) {
auto it_const = r_constraints_array.begin() + i;

it_const->GetDofList(dof_list, second_dof_list, r_current_process_info);
dofs_tmp_set.insert(dof_list.begin(), dof_list.end());
dofs_tmp_set.insert(second_dof_list.begin(), second_dof_list.end());
}

#pragma omp critical
{
dof_global_set.insert(dofs_tmp_set.begin(), dofs_tmp_set.end());
}
}

KRATOS_INFO_IF("ResidualBasedBlockBuilderAndSolver", ( this->GetEchoLevel() > 2)) << "Initializing ordered array filling\n" << std::endl;

DofsArrayType Doftemp;
BaseType::mDofSet = DofsArrayType();

Doftemp.reserve(dof_global_set.size());
for (auto it= dof_global_set.begin(); it!= dof_global_set.end(); it++)
{
Doftemp.push_back( *it );
}
Doftemp.Sort();

BaseType::mDofSet = Doftemp;

KRATOS_ERROR_IF(BaseType::mDofSet.size() == 0) << "No degrees of freedom!" << std::endl;

KRATOS_INFO_IF("ResidualBasedBlockBuilderAndSolver", ( this->GetEchoLevel() > 2)) << "Number of degrees of freedom:" << BaseType::mDofSet.size() << std::endl;

BaseType::mDofSetIsInitialized = true;

KRATOS_INFO_IF("ResidualBasedBlockBuilderAndSolver", ( this->GetEchoLevel() > 2 && rModelPart.GetCommunicator().MyPID() == 0)) << "Finished setting up the dofs" << std::endl;

#ifdef KRATOS_DEBUG
if (BaseType::GetCalculateReactionsFlag()) {
for (auto dof_iterator = BaseType::mDofSet.begin(); dof_iterator != BaseType::mDofSet.end(); ++dof_iterator) {
KRATOS_ERROR_IF_NOT(dof_iterator->HasReaction()) << "Reaction variable not set for the following : " <<std::endl
<< "Node : "<<dof_iterator->Id()<< std::endl
<< "Dof : "<<(*dof_iterator)<<std::endl<<"Not possible to calculate reactions."<<std::endl;
}
}
#endif

KRATOS_CATCH("");
}


void SetUpSystem(
ModelPart& rModelPart
) override
{
BaseType::mEquationSystemSize = BaseType::mDofSet.size();

IndexPartition<std::size_t>(BaseType::mDofSet.size()).for_each([&, this](std::size_t Index){
typename DofsArrayType::iterator dof_iterator = this->mDofSet.begin() + Index;
dof_iterator->SetEquationId(Index);
});
}

/
void ApplyDirichletConditions(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemMatrixType& rA,
TSystemVectorType& rDx,
TSystemVectorType& rb
) override
{
const std::size_t system_size = rA.size1();
Vector scaling_factors (system_size);

const auto it_dof_iterator_begin = BaseType::mDofSet.begin();

IndexPartition<std::size_t>(BaseType::mDofSet.size()).for_each([&](std::size_t Index){
auto it_dof_iterator = it_dof_iterator_begin + Index;
if (it_dof_iterator->IsFixed()) {
scaling_factors[Index] = 0.0;
} else {
scaling_factors[Index] = 1.0;
}
});

mScaleFactor = TSparseSpace::CheckAndCorrectZeroDiagonalValues(rModelPart.GetProcessInfo(), rA, rb, mScalingDiagonal);

double* Avalues = rA.value_data().begin();
std::size_t* Arow_indices = rA.index1_data().begin();
std::size_t* Acol_indices = rA.index2_data().begin();

IndexPartition<std::size_t>(system_size).for_each([&](std::size_t Index){
const std::size_t col_begin = Arow_indices[Index];
const std::size_t col_end = Arow_indices[Index+1];
const double k_factor = scaling_factors[Index];
if (k_factor == 0.0) {
for (std::size_t j = col_begin; j < col_end; ++j)
if (Acol_indices[j] != Index )
Avalues[j] = 0.0;

rb[Index] = 0.0;
} else {
for (std::size_t j = col_begin; j < col_end; ++j)
if(scaling_factors[ Acol_indices[j] ] == 0 )
Avalues[j] = 0.0;
}
});
}


void ApplyRHSConstraints(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemVectorType& rb
) override
{
KRATOS_TRY

if (rModelPart.MasterSlaveConstraints().size() != 0) {
BuildMasterSlaveConstraints(rModelPart);

TSystemMatrixType T_transpose_matrix(mT.size2(), mT.size1());
SparseMatrixMultiplicationUtility::TransposeMatrix<TSystemMatrixType, TSystemMatrixType>(T_transpose_matrix, mT, 1.0);

TSystemVectorType b_modified(rb.size());
TSparseSpace::Mult(T_transpose_matrix, rb, b_modified);
TSparseSpace::Copy(b_modified, rb);

IndexPartition<std::size_t>(mSlaveIds.size()).for_each([&](std::size_t Index){
const IndexType slave_equation_id = mSlaveIds[Index];
if (mInactiveSlaveDofs.find(slave_equation_id) == mInactiveSlaveDofs.end()) {
rb[slave_equation_id] = 0.0;
}
});
}

KRATOS_CATCH("")
}


void ApplyConstraints(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemMatrixType& rA,
TSystemVectorType& rb
) override
{
KRATOS_TRY

if (rModelPart.MasterSlaveConstraints().size() != 0) {
BuildMasterSlaveConstraints(rModelPart);

TSystemMatrixType T_transpose_matrix(mT.size2(), mT.size1());
SparseMatrixMultiplicationUtility::TransposeMatrix<TSystemMatrixType, TSystemMatrixType>(T_transpose_matrix, mT, 1.0);

TSystemVectorType b_modified(rb.size());
TSparseSpace::Mult(T_transpose_matrix, rb, b_modified);
TSparseSpace::Copy(b_modified, rb);

TSystemMatrixType auxiliar_A_matrix(mT.size2(), rA.size2());
SparseMatrixMultiplicationUtility::MatrixMultiplication(T_transpose_matrix, rA, auxiliar_A_matrix); 
T_transpose_matrix.resize(0, 0, false);                                                             

SparseMatrixMultiplicationUtility::MatrixMultiplication(auxiliar_A_matrix, mT, rA); 
auxiliar_A_matrix.resize(0, 0, false);                                              

const double max_diag = TSparseSpace::GetMaxDiagonal(rA);

IndexPartition<std::size_t>(mSlaveIds.size()).for_each([&](std::size_t Index){
const IndexType slave_equation_id = mSlaveIds[Index];
if (mInactiveSlaveDofs.find(slave_equation_id) == mInactiveSlaveDofs.end()) {
rA(slave_equation_id, slave_equation_id) = max_diag;
rb[slave_equation_id] = 0.0;
}
});
}

KRATOS_CATCH("")
}


void Clear() override
{
BaseType::Clear();

mSlaveIds.clear();
mMasterIds.clear();
mInactiveSlaveDofs.clear();
mT.resize(0,0,false);
mConstantVector.resize(0,false);
}


int Check(ModelPart& rModelPart) override
{
KRATOS_TRY

return 0;
KRATOS_CATCH("");
}


Parameters GetDefaultParameters() const override
{
Parameters default_parameters = Parameters(R"(
{
"name"                                 : "block_builder_and_solver",
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
return "block_builder_and_solver";
}



typename TSparseSpace::MatrixType& GetConstraintRelationMatrix() override
{
return mT;
}


typename TSparseSpace::VectorType& GetConstraintConstantVector() override
{
return mConstantVector;
}



std::string Info() const override
{
return "ResidualBasedBlockBuilderAndSolver";
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


TSystemMatrixType mT;                             
TSystemVectorType mConstantVector;                
std::vector<IndexType> mSlaveIds;                 
std::vector<IndexType> mMasterIds;                
std::unordered_set<IndexType> mInactiveSlaveDofs; 
double mScaleFactor = 1.0;                        

SCALING_DIAGONAL mScalingDiagonal = SCALING_DIAGONAL::CONSIDER_MAX_DIAGONAL; 
Flags mOptions;                                                              



void BuildRHSNoDirichlet(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemVectorType& b)
{
KRATOS_TRY

ElementsArrayType& pElements = rModelPart.Elements();

ConditionsArrayType& ConditionsArray = rModelPart.Conditions();

const ProcessInfo& CurrentProcessInfo = rModelPart.GetProcessInfo();

LocalSystemMatrixType LHS_Contribution = LocalSystemMatrixType(0, 0);
LocalSystemVectorType RHS_Contribution = LocalSystemVectorType(0);

Element::EquationIdVectorType EquationId;


const int nelements = static_cast<int>(pElements.size());
#pragma omp parallel firstprivate(nelements, RHS_Contribution, EquationId)
{
#pragma omp for schedule(guided, 512) nowait
for (int i=0; i<nelements; i++) {
typename ElementsArrayType::iterator it = pElements.begin() + i;
if(it->IsActive()) {
pScheme->CalculateRHSContribution(*it, RHS_Contribution, EquationId, CurrentProcessInfo);

AssembleRHS(b, RHS_Contribution, EquationId);
}
}

LHS_Contribution.resize(0, 0, false);
RHS_Contribution.resize(0, false);

const int nconditions = static_cast<int>(ConditionsArray.size());
#pragma omp for schedule(guided, 512)
for (int i = 0; i<nconditions; i++) {
auto it = ConditionsArray.begin() + i;
if(it->IsActive()) {
pScheme->CalculateRHSContribution(*it, RHS_Contribution, EquationId, CurrentProcessInfo);

AssembleRHS(b, RHS_Contribution, EquationId);
}
}
}

KRATOS_CATCH("")

}

virtual void ConstructMasterSlaveConstraintsStructure(ModelPart& rModelPart)
{
if (rModelPart.MasterSlaveConstraints().size() > 0) {
Timer::Start("ConstraintsRelationMatrixStructure");
const ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();

const auto it_const_begin = rModelPart.MasterSlaveConstraints().begin();
std::vector<std::unordered_set<IndexType>> indices(BaseType::mDofSet.size());

std::vector<LockObject> lock_array(indices.size());

#pragma omp parallel
{
Element::EquationIdVectorType slave_ids(3);
Element::EquationIdVectorType master_ids(3);
std::unordered_map<IndexType, std::unordered_set<IndexType>> temp_indices;

#pragma omp for schedule(guided, 512) nowait
for (int i_const = 0; i_const < static_cast<int>(rModelPart.MasterSlaveConstraints().size()); ++i_const) {
auto it_const = it_const_begin + i_const;
it_const->EquationIdVector(slave_ids, master_ids, r_current_process_info);

for (auto &id_i : slave_ids) {
temp_indices[id_i].insert(master_ids.begin(), master_ids.end());
}
}

for (auto& pair_temp_indices : temp_indices) {
lock_array[pair_temp_indices.first].lock();
indices[pair_temp_indices.first].insert(pair_temp_indices.second.begin(), pair_temp_indices.second.end());
lock_array[pair_temp_indices.first].unlock();
}
}

mSlaveIds.clear();
mMasterIds.clear();
for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
if (indices[i].size() == 0) 
mMasterIds.push_back(i);
else 
mSlaveIds.push_back(i);
indices[i].insert(i); 
}

const std::size_t nnz = block_for_each<SumReduction<std::size_t>>(indices, [](auto& rIndices) {return rIndices.size();});

mT = TSystemMatrixType(indices.size(), indices.size(), nnz);
mConstantVector.resize(indices.size(), false);

double *Tvalues = mT.value_data().begin();
IndexType *Trow_indices = mT.index1_data().begin();
IndexType *Tcol_indices = mT.index2_data().begin();

Trow_indices[0] = 0;
for (int i = 0; i < static_cast<int>(mT.size1()); i++)
Trow_indices[i + 1] = Trow_indices[i] + indices[i].size();

IndexPartition<std::size_t>(mT.size1()).for_each([&](std::size_t Index){
const IndexType row_begin = Trow_indices[Index];
const IndexType row_end = Trow_indices[Index + 1];
IndexType k = row_begin;
for (auto it = indices[Index].begin(); it != indices[Index].end(); ++it) {
Tcol_indices[k] = *it;
Tvalues[k] = 0.0;
k++;
}

indices[Index].clear(); 

std::sort(&Tcol_indices[row_begin], &Tcol_indices[row_end]);
});

mT.set_filled(indices.size() + 1, nnz);

Timer::Stop("ConstraintsRelationMatrixStructure");
}
}

virtual void BuildMasterSlaveConstraints(ModelPart& rModelPart)
{
KRATOS_TRY

TSparseSpace::SetToZero(mT);
TSparseSpace::SetToZero(mConstantVector);

const ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();

Matrix transformation_matrix = LocalSystemMatrixType(0, 0);
Vector constant_vector = LocalSystemVectorType(0);

Element::EquationIdVectorType slave_equation_ids, master_equation_ids;

const int number_of_constraints = static_cast<int>(rModelPart.MasterSlaveConstraints().size());

mInactiveSlaveDofs.clear();

#pragma omp parallel firstprivate(transformation_matrix, constant_vector, slave_equation_ids, master_equation_ids)
{
std::unordered_set<IndexType> auxiliar_inactive_slave_dofs;

#pragma omp for schedule(guided, 512)
for (int i_const = 0; i_const < number_of_constraints; ++i_const) {
auto it_const = rModelPart.MasterSlaveConstraints().begin() + i_const;
it_const->EquationIdVector(slave_equation_ids, master_equation_ids, r_current_process_info);

if (it_const->IsActive()) {
it_const->CalculateLocalSystem(transformation_matrix, constant_vector, r_current_process_info);

for (IndexType i = 0; i < slave_equation_ids.size(); ++i) {
const IndexType i_global = slave_equation_ids[i];

AssembleRowContribution(mT, transformation_matrix, i_global, i, master_equation_ids);

const double constant_value = constant_vector[i];
double& r_value = mConstantVector[i_global];
AtomicAdd(r_value, constant_value);
}
} else { 
auxiliar_inactive_slave_dofs.insert(slave_equation_ids.begin(), slave_equation_ids.end());
}
}

#pragma omp critical
{
mInactiveSlaveDofs.insert(auxiliar_inactive_slave_dofs.begin(), auxiliar_inactive_slave_dofs.end());
}
}

for (auto eq_id : mMasterIds) {
mConstantVector[eq_id] = 0.0;
mT(eq_id, eq_id) = 1.0;
}

for (auto eq_id : mInactiveSlaveDofs) {
mConstantVector[eq_id] = 0.0;
mT(eq_id, eq_id) = 1.0;
}

KRATOS_CATCH("")
}

virtual void ConstructMatrixStructure(
typename TSchemeType::Pointer pScheme,
TSystemMatrixType& A,
ModelPart& rModelPart)
{
Timer::Start("MatrixStructure");

const ProcessInfo& CurrentProcessInfo = rModelPart.GetProcessInfo();

const std::size_t equation_size = BaseType::mEquationSystemSize;

std::vector< LockObject > lock_array(equation_size);

std::vector<std::unordered_set<std::size_t> > indices(equation_size);

block_for_each(indices, [](std::unordered_set<std::size_t>& rIndices){
rIndices.reserve(40);
});

Element::EquationIdVectorType ids(3, 0);

block_for_each(rModelPart.Elements(), ids, [&](Element& rElem, Element::EquationIdVectorType& rIdsTLS){
pScheme->EquationId(rElem, rIdsTLS, CurrentProcessInfo);
for (std::size_t i = 0; i < rIdsTLS.size(); i++) {
lock_array[rIdsTLS[i]].lock();
auto& row_indices = indices[rIdsTLS[i]];
row_indices.insert(rIdsTLS.begin(), rIdsTLS.end());
lock_array[rIdsTLS[i]].unlock();
}
});

block_for_each(rModelPart.Conditions(), ids, [&](Condition& rCond, Element::EquationIdVectorType& rIdsTLS){
pScheme->EquationId(rCond, rIdsTLS, CurrentProcessInfo);
for (std::size_t i = 0; i < rIdsTLS.size(); i++) {
lock_array[rIdsTLS[i]].lock();
auto& row_indices = indices[rIdsTLS[i]];
row_indices.insert(rIdsTLS.begin(), rIdsTLS.end());
lock_array[rIdsTLS[i]].unlock();
}
});

if (rModelPart.MasterSlaveConstraints().size() != 0) {
struct TLS
{
Element::EquationIdVectorType master_ids = Element::EquationIdVectorType(3,0);
Element::EquationIdVectorType slave_ids = Element::EquationIdVectorType(3,0);
};
TLS tls;

block_for_each(rModelPart.MasterSlaveConstraints(), tls, [&](MasterSlaveConstraint& rConst, TLS& rTls){
rConst.EquationIdVector(rTls.slave_ids, rTls.master_ids, CurrentProcessInfo);

for (std::size_t i = 0; i < rTls.slave_ids.size(); i++) {
lock_array[rTls.slave_ids[i]].lock();
auto& row_indices = indices[rTls.slave_ids[i]];
row_indices.insert(rTls.slave_ids[i]);
lock_array[rTls.slave_ids[i]].unlock();
}

for (std::size_t i = 0; i < rTls.master_ids.size(); i++) {
lock_array[rTls.master_ids[i]].lock();
auto& row_indices = indices[rTls.master_ids[i]];
row_indices.insert(rTls.master_ids[i]);
lock_array[rTls.master_ids[i]].unlock();
}
});

}

lock_array = std::vector< LockObject >();

const std::size_t nnz = block_for_each<SumReduction<std::size_t>>(indices, [](auto& rIndices) {return rIndices.size();});

A = CompressedMatrixType(indices.size(), indices.size(), nnz);

double* Avalues = A.value_data().begin();
std::size_t* Arow_indices = A.index1_data().begin();
std::size_t* Acol_indices = A.index2_data().begin();

Arow_indices[0] = 0;
for (int i = 0; i < static_cast<int>(A.size1()); i++) {
Arow_indices[i+1] = Arow_indices[i] + indices[i].size();
}

IndexPartition<std::size_t>(A.size1()).for_each([&](std::size_t i){
const unsigned int row_begin = Arow_indices[i];
const unsigned int row_end = Arow_indices[i+1];
unsigned int k = row_begin;
for (auto it = indices[i].begin(); it != indices[i].end(); it++) {
Acol_indices[k] = *it;
Avalues[k] = 0.0;
k++;
}

indices[i].clear(); 

std::sort(&Acol_indices[row_begin], &Acol_indices[row_end]);

});

A.set_filled(indices.size()+1, nnz);

Timer::Stop("MatrixStructure");
}

void Assemble(
TSystemMatrixType& A,
TSystemVectorType& b,
const LocalSystemMatrixType& LHS_Contribution,
const LocalSystemVectorType& RHS_Contribution,
Element::EquationIdVectorType& EquationId
)
{
unsigned int local_size = LHS_Contribution.size1();

for (unsigned int i_local = 0; i_local < local_size; i_local++) {
unsigned int i_global = EquationId[i_local];

double& r_a = b[i_global];
const double& v_a = RHS_Contribution(i_local);
AtomicAdd(r_a, v_a);

AssembleRowContribution(A, LHS_Contribution, i_global, i_local, EquationId);
}
}

/
void AssignSettings(const Parameters ThisParameters) override
{
BaseType::AssignSettings(ThisParameters);

const std::string& r_diagonal_values_for_dirichlet_dofs = ThisParameters["diagonal_values_for_dirichlet_dofs"].GetString();

std::set<std::string> available_options_for_diagonal = {"no_scaling","use_max_diagonal","use_diagonal_norm","defined_in_process_info"};

if (available_options_for_diagonal.find(r_diagonal_values_for_dirichlet_dofs) == available_options_for_diagonal.end()) {
std::stringstream msg;
msg << "Currently prescribed diagonal values for dirichlet dofs : " << r_diagonal_values_for_dirichlet_dofs << "\n";
msg << "Admissible values for the diagonal scaling are : no_scaling, use_max_diagonal, use_diagonal_norm, or defined_in_process_info" << "\n";
KRATOS_ERROR << msg.str() << std::endl;
}

if (r_diagonal_values_for_dirichlet_dofs == "no_scaling") {
mScalingDiagonal = SCALING_DIAGONAL::NO_SCALING;
} else if (r_diagonal_values_for_dirichlet_dofs == "use_max_diagonal") {
mScalingDiagonal = SCALING_DIAGONAL::CONSIDER_MAX_DIAGONAL;
} else if (r_diagonal_values_for_dirichlet_dofs == "use_diagonal_norm") { 
mScalingDiagonal = SCALING_DIAGONAL::CONSIDER_NORM_DIAGONAL;
} else { 
mScalingDiagonal = SCALING_DIAGONAL::CONSIDER_PRESCRIBED_DIAGONAL;
}
mOptions.Set(SILENT_WARNINGS, ThisParameters["silent_warnings"].GetBool());
}





private:




inline void AddUnique(std::vector<std::size_t>& v, const std::size_t& candidate)
{
std::vector<std::size_t>::iterator i = v.begin();
std::vector<std::size_t>::iterator endit = v.end();
while (i != endit && (*i) != candidate) {
i++;
}
if (i == endit) {
v.push_back(candidate);
}
}

/



template<class TSparseSpace, class TDenseSpace, class TLinearSolver>
const Kratos::Flags ResidualBasedBlockBuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver>::SILENT_WARNINGS(Kratos::Flags::Create(0));


} 
