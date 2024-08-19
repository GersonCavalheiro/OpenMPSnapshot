
#pragma once

#include <unordered_set>



#include <Epetra_FECrsGraph.h>
#include <Epetra_IntVector.h>

#include "trilinos_space.h"
#include "solving_strategies/builder_and_solvers/builder_and_solver.h"
#include "utilities/timer.h"
#include "utilities/builtin_timer.h"

#if !defined(START_TIMER)
#define START_TIMER(label, rank) \
if (mrComm.MyPID() == rank)  \
Timer::Start(label);
#endif
#if !defined(STOP_TIMER)
#define STOP_TIMER(label, rank) \
if (mrComm.MyPID() == rank) \
Timer::Stop(label);
#endif

namespace Kratos {







template <class TSparseSpace,
class TDenseSpace,  
class TLinearSolver 
>
class TrilinosBlockBuilderAndSolver
: public BuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver> {
public:
KRATOS_CLASS_POINTER_DEFINITION(TrilinosBlockBuilderAndSolver);

typedef BuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver> BaseType;

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
typedef typename BaseType::ElementsContainerType ElementsContainerType;

typedef Epetra_MpiComm EpetraCommunicatorType;

typedef Node NodeType;
typedef typename NodeType::DofType DofType;
typedef DofType::Pointer DofPointerType;



TrilinosBlockBuilderAndSolver(EpetraCommunicatorType& rComm,
int GuessRowSize,
typename TLinearSolver::Pointer pNewLinearSystemSolver)
: BuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver>(pNewLinearSystemSolver),
mrComm(rComm),
mGuessRowSize(GuessRowSize)
{
}


~TrilinosBlockBuilderAndSolver() override = default;


TrilinosBlockBuilderAndSolver(const TrilinosBlockBuilderAndSolver& rOther) = delete;


TrilinosBlockBuilderAndSolver& operator=(const TrilinosBlockBuilderAndSolver& rOther) = delete;




void Build(typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemMatrixType& rA,
TSystemVectorType& rb) override
{
KRATOS_TRY

KRATOS_ERROR_IF(!pScheme) << "No scheme provided!" << std::endl;

TSparseSpace::SetToZero(*BaseType::mpReactionsVector);

LocalSystemMatrixType LHS_Contribution = LocalSystemMatrixType(0, 0);
LocalSystemVectorType RHS_Contribution = LocalSystemVectorType(0);

Element::EquationIdVectorType equation_ids_vector;
const ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();

BuiltinTimer build_timer;

for (auto it = rModelPart.Elements().ptr_begin(); it < rModelPart.Elements().ptr_end(); it++) {
const bool element_is_active = !((*it)->IsDefined(ACTIVE)) || (*it)->Is(ACTIVE);

if (element_is_active) {
pScheme->CalculateSystemContributions(**it, LHS_Contribution, RHS_Contribution, equation_ids_vector, r_current_process_info);

TSparseSpace::AssembleLHS(rA, LHS_Contribution, equation_ids_vector);
TSparseSpace::AssembleRHS(rb, RHS_Contribution, equation_ids_vector);
}
}

LHS_Contribution.resize(0, 0, false);
RHS_Contribution.resize(0, false);

for (auto it = rModelPart.Conditions().ptr_begin(); it < rModelPart.Conditions().ptr_end(); it++) {
const bool condition_is_active = !((*it)->IsDefined(ACTIVE)) || (*it)->Is(ACTIVE);
if (condition_is_active) {
pScheme->CalculateSystemContributions(**it, LHS_Contribution, RHS_Contribution, equation_ids_vector, r_current_process_info);

TSparseSpace::AssembleLHS(rA, LHS_Contribution, equation_ids_vector);
TSparseSpace::AssembleRHS(rb, RHS_Contribution, equation_ids_vector);
}
}

rA.GlobalAssemble();
rb.GlobalAssemble();

KRATOS_INFO_IF("TrilinosBlockBuilderAndSolver", BaseType::GetEchoLevel() >= 1) << "Build time: " << build_timer.ElapsedSeconds() << std::endl;

KRATOS_CATCH("")
}


void BuildLHS(typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemMatrixType& rA) override
{
KRATOS_TRY
TSparseSpace::SetToZero(*BaseType::mpReactionsVector);

LocalSystemMatrixType LHS_Contribution = LocalSystemMatrixType(0, 0);
LocalSystemVectorType RHS_Contribution = LocalSystemVectorType(0);

Element::EquationIdVectorType equation_ids_vector;
const ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();

BuiltinTimer build_timer;

for (auto it = rModelPart.Elements().ptr_begin(); it < rModelPart.Elements().ptr_end(); it++) {
pScheme->CalculateLHSContribution(**it, LHS_Contribution, equation_ids_vector, r_current_process_info);

TSparseSpace::AssembleLHS(rA, LHS_Contribution, equation_ids_vector);
}

LHS_Contribution.resize(0, 0, false);

for (auto it = rModelPart.Conditions().ptr_begin(); it < rModelPart.Conditions().ptr_end(); it++) {
pScheme->CalculateLHSContribution(**it, LHS_Contribution, equation_ids_vector, r_current_process_info);

TSparseSpace::AssembleLHS(rA, LHS_Contribution, equation_ids_vector);
}

rA.GlobalAssemble();

KRATOS_INFO_IF("TrilinosBlockBuilderAndSolver", BaseType::GetEchoLevel() >= 1) << "Build time LHS: " << build_timer.ElapsedSeconds() << std::endl;

KRATOS_CATCH("")
}


void BuildLHS_CompleteOnFreeRows(typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemMatrixType& A) override
{
KRATOS_ERROR << "Method BuildLHS_CompleteOnFreeRows not implemented in "
"Trilinos Builder And Solver"
<< std::endl;
}


void SystemSolveWithPhysics(TSystemMatrixType& rA,
TSystemVectorType& rDx,
TSystemVectorType& rb,
ModelPart& rModelPart)
{
KRATOS_TRY

double norm_b;
if (TSparseSpace::Size(rb) != 0)
norm_b = TSparseSpace::TwoNorm(rb);
else
norm_b = 0.00;

if (norm_b != 0.00) {
if (BaseType::mpLinearSystemSolver->AdditionalPhysicalDataIsNeeded())
BaseType::mpLinearSystemSolver->ProvideAdditionalData(
rA, rDx, rb, BaseType::mDofSet, rModelPart);

BaseType::mpLinearSystemSolver->Solve(rA, rDx, rb);
}
else {
TSparseSpace::SetToZero(rDx);
KRATOS_WARNING(
"TrilinosResidualBasedBlockBuilderAndSolver")
<< "ATTENTION! setting the RHS to zero!" << std::endl;
}

KRATOS_INFO_IF("TrilinosResidualBasedBlockBuilderAndSolver", (BaseType::GetEchoLevel() > 1))
<< *(BaseType::mpLinearSystemSolver) << std::endl;

KRATOS_CATCH("")
}


void BuildAndSolve(typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemMatrixType& rA,
TSystemVectorType& rDx,
TSystemVectorType& rb) override
{
KRATOS_TRY

START_TIMER("Build", 0)

Build(pScheme, rModelPart, rA, rb);

STOP_TIMER("Build", 0)

ApplyDirichletConditions(pScheme, rModelPart, rA, rDx, rb);

KRATOS_INFO_IF("TrilinosBlockBuilderAndSolver", BaseType::GetEchoLevel() == 3)
<< "\nBefore the solution of the system"
<< "\nSystem Matrix = " << rA << "\nunknowns vector = " << rDx
<< "\nRHS vector = " << rb << std::endl;

START_TIMER("Solve", 0)

BuiltinTimer solve_timer;

SystemSolveWithPhysics(rA, rDx, rb, rModelPart);

KRATOS_INFO_IF("TrilinosBlockBuilderAndSolver", BaseType::GetEchoLevel() >=1) << "System solve time: " << solve_timer.ElapsedSeconds() << std::endl;

STOP_TIMER("Solve", 0)

KRATOS_INFO_IF("TrilinosBlockBuilderAndSolver", BaseType::GetEchoLevel() == 3)
<< "\nAfter the solution of the system"
<< "\nSystem Matrix = " << rA << "\nUnknowns vector = " << rDx
<< "\nRHS vector = " << rb << std::endl;
KRATOS_CATCH("")
}


void BuildRHSAndSolve(typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemMatrixType& rA,
TSystemVectorType& rDx,
TSystemVectorType& rb) override
{
KRATOS_TRY

BuildRHS(pScheme, rModelPart, rb);

BuiltinTimer solve_timer;

SystemSolveWithPhysics(rA, rDx, rb, rModelPart);

KRATOS_INFO_IF("TrilinosBlockBuilderAndSolver", BaseType::GetEchoLevel() >=1) << "System solve time: " << solve_timer.ElapsedSeconds() << std::endl;

KRATOS_CATCH("")
}


void BuildRHS(typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemVectorType& rb) override
{
KRATOS_TRY

START_TIMER("BuildRHS ", 0)
TSparseSpace::SetToZero(*BaseType::mpReactionsVector);

LocalSystemMatrixType LHS_Contribution = LocalSystemMatrixType(0, 0);
LocalSystemVectorType RHS_Contribution = LocalSystemVectorType(0);

Element::EquationIdVectorType equation_ids_vector;
const ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();

for (auto it = rModelPart.Elements().ptr_begin(); it < rModelPart.Elements().ptr_end(); it++) {
pScheme->CalculateRHSContribution(**it, RHS_Contribution, equation_ids_vector, r_current_process_info);

TSparseSpace::AssembleRHS(rb, RHS_Contribution, equation_ids_vector);
}

RHS_Contribution.resize(0, false);

for (auto it = rModelPart.Conditions().ptr_begin(); it < rModelPart.Conditions().ptr_end(); it++) {
pScheme->CalculateRHSContribution(**it, RHS_Contribution, equation_ids_vector, r_current_process_info);

TSparseSpace::AssembleRHS(rb, RHS_Contribution, equation_ids_vector);
}

rb.GlobalAssemble();

STOP_TIMER("BuildRHS ", 0)

KRATOS_CATCH("")
}


void SetUpDofSet(
typename TSchemeType::Pointer pScheme, 
ModelPart& rModelPart
) override
{
KRATOS_TRY

using DofsVectorType = Element::DofsVectorType;

DofsVectorType dof_list;
const ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();

DofsArrayType temp_dofs_array;
IndexType guess_num_dofs = rModelPart.GetCommunicator().LocalMesh().NumberOfNodes() * 3;
temp_dofs_array.reserve(guess_num_dofs);
BaseType::mDofSet = DofsArrayType();

ElementsArrayType& r_elements_array = rModelPart.GetCommunicator().LocalMesh().Elements();
for (auto it_elem = r_elements_array.ptr_begin(); it_elem != r_elements_array.ptr_end(); ++it_elem) {
pScheme->GetDofList(**it_elem, dof_list, r_current_process_info);
for (auto i_dof = dof_list.begin(); i_dof != dof_list.end(); ++i_dof)
temp_dofs_array.push_back(*i_dof);
}

auto& r_conditions_array = rModelPart.GetCommunicator().LocalMesh().Conditions();
for (auto it_cond = r_conditions_array.ptr_begin(); it_cond != r_conditions_array.ptr_end(); ++it_cond) {
pScheme->GetDofList(**it_cond, dof_list, r_current_process_info);
for (auto i_dof = dof_list.begin(); i_dof != dof_list.end(); ++i_dof)
temp_dofs_array.push_back(*i_dof);
}

temp_dofs_array.Unique();
BaseType::mDofSet = temp_dofs_array;

KRATOS_ERROR_IF(rModelPart.GetCommunicator().GetDataCommunicator().SumAll(BaseType::mDofSet.size()) == 0) << "No degrees of freedom!";

#ifdef KRATOS_DEBUG
if (BaseType::GetCalculateReactionsFlag()) {
for (auto dof_iterator = BaseType::mDofSet.begin();
dof_iterator != BaseType::mDofSet.end(); ++dof_iterator) {
KRATOS_ERROR_IF_NOT(dof_iterator->HasReaction())
<< "Reaction variable not set for the following : " << std::endl
<< "Node : " << dof_iterator->Id() << std::endl
<< "Dof : " << (*dof_iterator) << std::endl
<< "Not possible to calculate reactions." << std::endl;
}
}
#endif
BaseType::mDofSetIsInitialized = true;

KRATOS_CATCH("")
}


void SetUpSystem(ModelPart& rModelPart) override
{
int free_size = 0;
auto& r_comm = rModelPart.GetCommunicator();
const auto& r_data_comm = r_comm.GetDataCommunicator();
int current_rank = r_comm.MyPID();

for (const auto& r_dof : BaseType::mDofSet)
if (r_dof.GetSolutionStepValue(PARTITION_INDEX) == current_rank)
free_size++;

int free_offset;
int global_size;

free_offset = r_data_comm.ScanSum(free_size);

global_size = r_data_comm.SumAll(free_size);

free_offset -= free_size;

for (auto& r_dof : BaseType::mDofSet)
if (r_dof.GetSolutionStepValue(PARTITION_INDEX) == current_rank)
r_dof.SetEquationId(free_offset++);

BaseType::mEquationSystemSize = global_size;
mLocalSystemSize = free_size;
KRATOS_INFO_IF_ALL_RANKS("TrilinosBlockBuilderAndSolver", BaseType::GetEchoLevel() > 1)
<< "\n    BaseType::mEquationSystemSize = " << BaseType::mEquationSystemSize
<< "\n    mLocalSystemSize = " << mLocalSystemSize
<< "\n    free_offset = " << free_offset << std::endl;

mFirstMyId = free_offset - mLocalSystemSize;
mLastMyId = mFirstMyId + mLocalSystemSize;

r_comm.SynchronizeDofs();
}


void ResizeAndInitializeVectors(typename TSchemeType::Pointer pScheme,
TSystemMatrixPointerType& rpA,
TSystemVectorPointerType& rpDx,
TSystemVectorPointerType& rpb,
ModelPart& rModelPart) override
{
KRATOS_TRY

if (rpA == nullptr || TSparseSpace::Size1(*rpA) == 0 ||
BaseType::GetReshapeMatrixFlag() == true) { 
IndexType number_of_local_dofs = mLastMyId - mFirstMyId;
int temp_size = number_of_local_dofs;
if (temp_size < 1000) {
temp_size = 1000;
}
std::vector<int> temp(temp_size, 0);

auto& r_elements_array = rModelPart.Elements();
auto& r_conditions_array = rModelPart.Conditions();

for (IndexType i = 0; i != number_of_local_dofs; i++) {
temp[i] = mFirstMyId + i;
}
Epetra_Map my_map(-1, number_of_local_dofs, temp.data(), 0, mrComm);

Epetra_FECrsGraph Agraph(Copy, my_map, mGuessRowSize);
Element::EquationIdVectorType equation_ids_vector;
const ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();

for (auto it_elem = r_elements_array.ptr_begin(); it_elem != r_elements_array.ptr_end(); ++it_elem) {
pScheme->EquationId(**it_elem, equation_ids_vector, r_current_process_info);

IndexType num_active_indices = 0;
for (IndexType i = 0; i < equation_ids_vector.size(); i++) {
temp[num_active_indices] = equation_ids_vector[i];
num_active_indices += 1;
}

if (num_active_indices != 0) {
const int ierr = Agraph.InsertGlobalIndices(num_active_indices, temp.data(), num_active_indices, temp.data());
KRATOS_ERROR_IF(ierr < 0) << ": Epetra failure in Graph.InsertGlobalIndices. Error code: " << ierr << std::endl;
}
std::fill(temp.begin(), temp.end(), 0);
}

for (auto it_cond = r_conditions_array.ptr_begin(); it_cond != r_conditions_array.ptr_end(); ++it_cond) {
pScheme->EquationId(**it_cond, equation_ids_vector, r_current_process_info);

IndexType num_active_indices = 0;
for (IndexType i = 0; i < equation_ids_vector.size(); i++) {
temp[num_active_indices] = equation_ids_vector[i];
num_active_indices += 1;
}

if (num_active_indices != 0) {
const int ierr = Agraph.InsertGlobalIndices(num_active_indices, temp.data(), num_active_indices, temp.data());
KRATOS_ERROR_IF(ierr < 0) << ": Epetra failure in Graph.InsertGlobalIndices. Error code: " << ierr << std::endl;
}
std::fill(temp.begin(), temp.end(), 0);
}

const int ierr = Agraph.GlobalAssemble();
KRATOS_ERROR_IF(ierr < 0) << ": Epetra failure in Graph.InsertGlobalIndices. Error code: " << ierr << std::endl;

TSystemMatrixPointerType p_new_A = TSystemMatrixPointerType(new TSystemMatrixType(Copy, Agraph));
rpA.swap(p_new_A);

if (rpb == nullptr || TSparseSpace::Size(*rpb) != BaseType::mEquationSystemSize) {
TSystemVectorPointerType p_new_b = TSystemVectorPointerType(new TSystemVectorType(my_map));
rpb.swap(p_new_b);
}
if (rpDx == nullptr || TSparseSpace::Size(*rpDx) != BaseType::mEquationSystemSize) {
TSystemVectorPointerType p_new_Dx = TSystemVectorPointerType(new TSystemVectorType(my_map));
rpDx.swap(p_new_Dx);
}
if (BaseType::mpReactionsVector == nullptr) { 
TSystemVectorPointerType pNewReactionsVector = TSystemVectorPointerType(new TSystemVectorType(my_map));
BaseType::mpReactionsVector.swap(pNewReactionsVector);
}
} else if (BaseType::mpReactionsVector == nullptr && this->mCalculateReactionsFlag) {
TSystemVectorPointerType pNewReactionsVector =
TSystemVectorPointerType(new TSystemVectorType(rpDx->Map()));
BaseType::mpReactionsVector.swap(pNewReactionsVector);
} else {
if (TSparseSpace::Size1(*rpA) == 0 ||
TSparseSpace::Size1(*rpA) != BaseType::mEquationSystemSize ||
TSparseSpace::Size2(*rpA) != BaseType::mEquationSystemSize) {
KRATOS_ERROR << "It should not come here resizing is not allowed this way!!!!!!!! ... " << std::endl;
}
}

KRATOS_CATCH("")
}

/
void ApplyDirichletConditions(typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemMatrixType& rA,
TSystemVectorType& rDx,
TSystemVectorType& rb) override
{
KRATOS_TRY

std::vector<int> global_ids(BaseType::mDofSet.size());
std::vector<int> is_dirichlet(BaseType::mDofSet.size());

IndexType i = 0;
for (const auto& dof : BaseType::mDofSet) {
const int global_id = dof.EquationId();
global_ids[i] = global_id;
is_dirichlet[i] = dof.IsFixed();
++i;
}

Epetra_Map localmap(-1, global_ids.size(), global_ids.data(), 0, rA.Comm());
Epetra_IntVector fixed_local(Copy, localmap, is_dirichlet.data());

Epetra_Import dirichlet_importer(rA.ColMap(), fixed_local.Map());

Epetra_IntVector fixed(rA.ColMap());

int ierr = fixed.Import(fixed_local, dirichlet_importer, Insert);
if (ierr != 0)
KRATOS_ERROR << "Epetra failure found";

for (int i = 0; i < rA.NumMyRows(); i++) {
int numEntries; 
double* vals;   
int* cols;      
rA.ExtractMyRowView(i, numEntries, vals, cols);

int row_gid = rA.RowMap().GID(i);
int row_lid = localmap.LID(row_gid);

if (fixed_local[row_lid] == 0) 
{
for (int j = 0; j < numEntries; j++) {
if (fixed[cols[j]] == true)
vals[j] = 0.0;
}
}
else 
{
rb[0][i] = 0.0; 

for (int j = 0; j < numEntries; j++) {
int col_gid = rA.ColMap().GID(cols[j]);
if (col_gid != row_gid)
vals[j] = 0.0;
}
}
}

KRATOS_CATCH("");
}




protected:


EpetraCommunicatorType& mrComm;
int mGuessRowSize;
IndexType mLocalSystemSize;
int mFirstMyId;
int mLastMyId;






private:




void AssembleLHS_CompleteOnFreeRows(TSystemMatrixType& rA,
LocalSystemMatrixType& rLHS_Contribution,
Element::EquationIdVectorType& rEquationId)
{
KRATOS_ERROR << "This method is not implemented for Trilinos";
}




}; 




} 
