
#if !defined(KRATOS_RESIDUAL_BASED_ELIMINATION_BUILDER_AND_SOLVER )
#define  KRATOS_RESIDUAL_BASED_ELIMINATION_BUILDER_AND_SOLVER


#include <set>
#include <unordered_set>


#ifdef KRATOS_SMP_OPENMP
#include <omp.h>
#endif


#include "utilities/timer.h"
#include "includes/define.h"
#include "includes/key_hash.h"
#include "solving_strategies/builder_and_solvers/builder_and_solver.h"
#include "includes/model_part.h"
#include "utilities/builtin_timer.h"
#include "utilities/atomic_utilities.h"
#include "spaces/ublas_space.h"

namespace Kratos
{







template<class TSparseSpace,
class TDenseSpace, 
class TLinearSolver 
>
class ResidualBasedEliminationBuilderAndSolver
: public BuilderAndSolver< TSparseSpace, TDenseSpace, TLinearSolver >
{
public:

KRATOS_CLASS_POINTER_DEFINITION(ResidualBasedEliminationBuilderAndSolver);

typedef BuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver> BaseType;

typedef ResidualBasedEliminationBuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver> ClassType;

typedef typename BaseType::SizeType SizeType;
typedef typename BaseType::IndexType IndexType;
typedef typename BaseType::TSchemeType TSchemeType;
typedef typename BaseType::TDataType TDataType;
typedef typename BaseType::DofsArrayType DofsArrayType;
typedef typename BaseType::TSystemMatrixType TSystemMatrixType;
typedef typename BaseType::TSystemVectorType TSystemVectorType;
typedef typename BaseType::LocalSystemVectorType LocalSystemVectorType;
typedef typename BaseType::LocalSystemMatrixType LocalSystemMatrixType;
typedef typename BaseType::TSystemMatrixPointerType TSystemMatrixPointerType;
typedef typename BaseType::TSystemVectorPointerType TSystemVectorPointerType;

typedef Element::EquationIdVectorType EquationIdVectorType;
typedef Element::DofsVectorType DofsVectorType;

typedef Node NodeType;

typedef typename BaseType::NodesArrayType NodesArrayType;
typedef typename BaseType::ElementsArrayType ElementsArrayType;
typedef typename BaseType::ConditionsArrayType ConditionsArrayType;
typedef typename BaseType::ElementsContainerType ElementsContainerType;



explicit ResidualBasedEliminationBuilderAndSolver() : BaseType()
{
}


explicit ResidualBasedEliminationBuilderAndSolver(
typename TLinearSolver::Pointer pNewLinearSystemSolver,
Parameters ThisParameters
) : BaseType(pNewLinearSystemSolver)
{
ThisParameters = this->ValidateAndAssignParameters(ThisParameters, this->GetDefaultParameters());
this->AssignSettings(ThisParameters);
}


explicit ResidualBasedEliminationBuilderAndSolver(
typename TLinearSolver::Pointer pNewLinearSystemSolver)
: BaseType(pNewLinearSystemSolver)
{
}


~ResidualBasedEliminationBuilderAndSolver() override
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
TSystemMatrixType& rA,
TSystemVectorType& rb
) override
{
KRATOS_TRY

KRATOS_ERROR_IF(!pScheme) << "No scheme provided!" << std::endl;

ElementsArrayType& r_elements_array = rModelPart.Elements();

ConditionsArrayType& r_conditions_array = rModelPart.Conditions();

const int nelements = static_cast<int>(r_elements_array.size());

const int nconditions = static_cast<int>(r_conditions_array.size());

const ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();
const auto it_elem_begin = r_elements_array.begin();
const auto it_cond_begin = r_conditions_array.begin();

LocalSystemMatrixType LHS_Contribution = LocalSystemMatrixType(0, 0);
LocalSystemVectorType RHS_Contribution = LocalSystemVectorType(0);

EquationIdVectorType equation_id;

const auto timer = BuiltinTimer();

#pragma omp parallel firstprivate(LHS_Contribution, RHS_Contribution, equation_id )
{
#pragma omp for schedule(guided, 512) nowait
for (int k = 0; k < nelements; ++k) {
auto it_elem = it_elem_begin + k;

if (it_elem->IsActive()) {
pScheme->CalculateSystemContributions(*it_elem, LHS_Contribution, RHS_Contribution, equation_id, r_current_process_info);

#ifdef USE_LOCKS_IN_ASSEMBLY
Assemble(rA, rb, LHS_Contribution, RHS_Contribution, equation_id, mLockArray);
#else
Assemble(rA, rb, LHS_Contribution, RHS_Contribution, equation_id);
#endif
}
}

#pragma omp for schedule(guided, 512)
for (int k = 0; k < nconditions; ++k) {
auto it_cond = it_cond_begin + k;

if (it_cond->IsActive()) {
pScheme->CalculateSystemContributions(*it_cond, LHS_Contribution, RHS_Contribution, equation_id, r_current_process_info);

#ifdef USE_LOCKS_IN_ASSEMBLY
Assemble(rA, rb, LHS_Contribution, RHS_Contribution, equation_id, mLockArray);
#else
Assemble(rA, rb, LHS_Contribution, RHS_Contribution, equation_id);
#endif
}
}
}
KRATOS_INFO_IF("ResidualBasedEliminationBuilderAndSolver", this->GetEchoLevel() >=1) << "System build time: " << timer.ElapsedSeconds() << std::endl;


KRATOS_INFO_IF("ResidualBasedEliminationBuilderAndSolver", this->GetEchoLevel() > 2) << "Finished building" << std::endl;


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

ElementsArrayType& r_elements_array = rModelPart.Elements();

ConditionsArrayType& r_conditions_array = rModelPart.Conditions();

const int nelements = static_cast<int>(r_elements_array.size());

const int nconditions = static_cast<int>(r_conditions_array.size());

const ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();
const auto it_elem_begin = r_elements_array.begin();
const auto it_cond_begin = r_conditions_array.begin();

TSparseSpace::SetToZero(*(BaseType::mpReactionsVector));

LocalSystemMatrixType LHS_Contribution = LocalSystemMatrixType(0, 0);

EquationIdVectorType equation_id;

#pragma omp parallel firstprivate(LHS_Contribution, equation_id )
{
#pragma omp for schedule(guided, 512) nowait
for (int k = 0; k < nelements; ++k) {
auto it_elem = it_elem_begin + k;

if (it_elem->IsActive()) {
pScheme->CalculateLHSContribution(*it_elem, LHS_Contribution, equation_id, r_current_process_info);

AssembleLHS(rA, LHS_Contribution, equation_id);
}
}

#pragma omp for schedule(guided, 512)
for (int k = 0; k < nconditions; ++k) {
auto it_cond = it_cond_begin + k;

if (it_cond->IsActive()) {
pScheme->CalculateLHSContribution(*it_cond, LHS_Contribution, equation_id, r_current_process_info);

AssembleLHS(rA, LHS_Contribution, equation_id);
}
}
}

KRATOS_CATCH("")
}


void BuildLHS_CompleteOnFreeRows(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemMatrixType& rA
) override
{
KRATOS_TRY

KRATOS_ERROR_IF(!pScheme) << "No scheme provided!" << std::endl;

ElementsArrayType& r_elements_array = rModelPart.Elements();

ConditionsArrayType& r_conditions_array = rModelPart.Conditions();

const int nelements = static_cast<int>(r_elements_array.size());

const int nconditions = static_cast<int>(r_conditions_array.size());

const ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();
const auto it_elem_begin = r_elements_array.begin();
const auto it_cond_begin = r_conditions_array.begin();

TSparseSpace::SetToZero(*(BaseType::mpReactionsVector));

LocalSystemMatrixType LHS_Contribution = LocalSystemMatrixType(0, 0);

EquationIdVectorType equation_id;

#pragma omp parallel firstprivate(LHS_Contribution, equation_id )
{
#pragma omp for schedule(guided, 512) nowait
for (int k = 0; k < nelements; ++k) {
auto it_elem = it_elem_begin + k;

if (it_elem->IsActive()) {
pScheme->CalculateLHSContribution(*it_elem, LHS_Contribution, equation_id, r_current_process_info);

AssembleLHSCompleteOnFreeRows(rA, LHS_Contribution, equation_id);
}
}

#pragma omp for schedule(guided, 512)
for (int k = 0; k < nconditions; ++k) {
auto it_cond = it_cond_begin + k;

if (it_cond->IsActive()) {
pScheme->CalculateLHSContribution(*it_cond, LHS_Contribution, equation_id, r_current_process_info);

AssembleLHSCompleteOnFreeRows(rA, LHS_Contribution, equation_id);
}
}
}

KRATOS_CATCH("")
}


void SystemSolve(
TSystemMatrixType& rA,
TSystemVectorType& rDx,
TSystemVectorType& rb
) override
{
KRATOS_TRY

double norm_b;
if (TSparseSpace::Size(rb) != 0) {
norm_b = TSparseSpace::TwoNorm(rb);
} else {
norm_b = 0.0;
}

if (norm_b != 0.0) {
BaseType::mpLinearSystemSolver->Solve(rA, rDx, rb);
} else
TSparseSpace::SetToZero(rDx);

KRATOS_INFO_IF("ResidualBasedEliminationBuilderAndSolver", this->GetEchoLevel() > 1) << *(BaseType::mpLinearSystemSolver) << std::endl;

KRATOS_CATCH("")

}


void SystemSolveWithPhysics(
TSystemMatrixType& rA,
TSystemVectorType& rDx,
TSystemVectorType& rb,
ModelPart& rModelPart
)
{
KRATOS_TRY

double norm_b;
if (TSparseSpace::Size(rb) != 0) {
norm_b = TSparseSpace::TwoNorm(rb);
} else {
norm_b = 0.0;
}

if (norm_b != 0.0) {
if(BaseType::mpLinearSystemSolver->AdditionalPhysicalDataIsNeeded() )
BaseType::mpLinearSystemSolver->ProvideAdditionalData(rA, rDx, rb, BaseType::mDofSet, rModelPart);

BaseType::mpLinearSystemSolver->Solve(rA, rDx, rb);
} else {
TSparseSpace::SetToZero(rDx);
KRATOS_WARNING_IF("ResidualBasedEliminationBuilderAndSolver", rModelPart.GetCommunicator().MyPID() == 0) << "ATTENTION! setting the RHS to zero!" << std::endl;
}

KRATOS_INFO_IF("ResidualBasedEliminationBuilderAndSolver", this->GetEchoLevel() > 1 && rModelPart.GetCommunicator().MyPID() == 0) << *(BaseType::mpLinearSystemSolver) << std::endl;

KRATOS_CATCH("")

}


void BuildAndSolve(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemMatrixType& rA,
TSystemVectorType& rDx,
TSystemVectorType& rb
) override
{
KRATOS_TRY

Timer::Start("Build");

Build(pScheme, rModelPart, rA, rb);

Timer::Stop("Build");

ApplyDirichletConditions(pScheme, rModelPart, rA, rDx, rb);

KRATOS_INFO_IF("ResidualBasedEliminationBuilderAndSolver", ( this->GetEchoLevel() == 3)) << "Before the solution of the system" << "\nSystem Matrix = " << rA << "\nUnknowns vector = " << rDx << "\nRHS vector = " << rb << std::endl;

const auto timer = BuiltinTimer();
Timer::Start("Solve");

SystemSolveWithPhysics(rA, rDx, rb, rModelPart);

Timer::Stop("Solve");
KRATOS_INFO_IF("ResidualBasedEliminationBuilderAndSolver", this->GetEchoLevel() >=1) << "System solve time: " << timer.ElapsedSeconds() << std::endl;


KRATOS_INFO_IF("ResidualBasedEliminationBuilderAndSolver", ( this->GetEchoLevel() == 3)) << "After the solution of the system" << "\nSystem Matrix = " << rA << "\nUnknowns vector = " << rDx << "\nRHS vector = " << rb << std::endl;

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
SystemSolve(rA, rDx, rb);

KRATOS_CATCH("")
}


void BuildRHS(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemVectorType& rb
) override
{
KRATOS_TRY

if(BaseType::mCalculateReactionsFlag) {
TSparseSpace::SetToZero(*(BaseType::mpReactionsVector));
}

ElementsArrayType& r_elements_array = rModelPart.Elements();

ConditionsArrayType& r_conditions_array = rModelPart.Conditions();

const ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();

LocalSystemVectorType RHS_Contribution = LocalSystemVectorType(0);

EquationIdVectorType equation_id;

#pragma omp parallel firstprivate( RHS_Contribution, equation_id)
{
const auto it_elem_begin = r_elements_array.begin();
const int nelements = static_cast<int>(r_elements_array.size());
#pragma omp for schedule(guided, 512) nowait
for (int i = 0; i < nelements; ++i) {
auto it_elem = it_elem_begin + i;

if (it_elem->IsActive()) {
pScheme->CalculateRHSContribution(*it_elem, RHS_Contribution, equation_id, r_current_process_info);

AssembleRHS(rb, RHS_Contribution, equation_id);
}
}

const auto it_cond_begin = r_conditions_array.begin();
const int nconditions = static_cast<int>(r_conditions_array.size());
#pragma omp for schedule(guided, 512)
for (int i = 0; i < nconditions; ++i) {
auto it_cond = it_cond_begin + i;
if (it_cond->IsActive()) {
pScheme->CalculateRHSContribution(*it_cond, RHS_Contribution, equation_id, r_current_process_info);

AssembleRHS(rb, RHS_Contribution, equation_id);
}
}
}

KRATOS_CATCH("")
}


void SetUpDofSet(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart
) override
{
KRATOS_TRY;

KRATOS_INFO_IF("ResidualBasedEliminationBuilderAndSolver", this->GetEchoLevel() > 1 && rModelPart.GetCommunicator().MyPID() == 0) << "Setting up the dofs" << std::endl;

ElementsArrayType& r_elements_array = rModelPart.Elements();
const int nelements = static_cast<int>(r_elements_array.size());

DofsVectorType elemental_dof_list;

const ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();

SizeType nthreads = ParallelUtilities::GetNumThreads();

typedef std::unordered_set < NodeType::DofType::Pointer, DofPointerHasher>  set_type;

std::vector<set_type> dofs_aux_list(nthreads);

for (int i = 0; i < static_cast<int>(nthreads); ++i) {
dofs_aux_list[i].reserve(nelements);
}

IndexPartition<std::size_t>(nelements).for_each(elemental_dof_list, [&](std::size_t Index, DofsVectorType& tls_elemental_dof_list){
auto it_elem = r_elements_array.begin() + Index;
const IndexType this_thread_id = OpenMPUtils::ThisThread();

pScheme->GetDofList(*it_elem, tls_elemental_dof_list, r_current_process_info);

dofs_aux_list[this_thread_id].insert(tls_elemental_dof_list.begin(), tls_elemental_dof_list.end());
});

ConditionsArrayType& r_conditions_array = rModelPart.Conditions();
const int nconditions = static_cast<int>(r_conditions_array.size());

IndexPartition<std::size_t>(nconditions).for_each(elemental_dof_list, [&](std::size_t Index, DofsVectorType& tls_elemental_dof_list){
auto it_cond = r_conditions_array.begin() + Index;
const IndexType this_thread_id = OpenMPUtils::ThisThread();

pScheme->GetDofList(*it_cond, tls_elemental_dof_list, r_current_process_info);
dofs_aux_list[this_thread_id].insert(tls_elemental_dof_list.begin(), tls_elemental_dof_list.end());
});

SizeType old_max = nthreads;
SizeType new_max = ceil(0.5*static_cast<double>(old_max));
while (new_max >= 1 && new_max != old_max) {
IndexPartition<std::size_t>(new_max).for_each([&](std::size_t Index){
if (Index + new_max < old_max) {
dofs_aux_list[Index].insert(dofs_aux_list[Index + new_max].begin(), dofs_aux_list[Index + new_max].end());
dofs_aux_list[Index + new_max].clear();
}
});

old_max = new_max;
new_max = ceil(0.5*static_cast<double>(old_max));
}

DofsArrayType dof_temp;
BaseType::mDofSet = DofsArrayType();

dof_temp.reserve(dofs_aux_list[0].size());
for (auto it = dofs_aux_list[0].begin(); it != dofs_aux_list[0].end(); ++it) {
dof_temp.push_back(*it);
}
dof_temp.Sort();

BaseType::mDofSet = dof_temp;

KRATOS_ERROR_IF(BaseType::mDofSet.size() == 0) << "No degrees of freedom!" << std::endl;

BaseType::mDofSetIsInitialized = true;

KRATOS_INFO_IF("ResidualBasedEliminationBuilderAndSolver", this->GetEchoLevel() > 2 && rModelPart.GetCommunicator().MyPID() == 0) << "Finished setting up the dofs" << std::endl;

#ifdef USE_LOCKS_IN_ASSEMBLY
if (mLockArray.size() != 0) {
for (int i = 0; i < static_cast<int>(mLockArray.size()); i++)
omp_destroy_lock(&mLockArray[i]);
}

mLockArray.resize(BaseType::mDofSet.size());

for (int i = 0; i < static_cast<int>(mLockArray.size()); i++)
omp_init_lock(&mLockArray[i]);
#endif

#ifdef KRATOS_DEBUG
if(BaseType::GetCalculateReactionsFlag()) {
for(auto dof_iterator = BaseType::mDofSet.begin(); dof_iterator != BaseType::mDofSet.end(); ++dof_iterator) {
KRATOS_ERROR_IF_NOT(dof_iterator->HasReaction()) << "Reaction variable not set for the following : " << std::endl
<< "Node : " << dof_iterator->Id() << std::endl
<< "Dof : " << (*dof_iterator) << std::endl << "Not possible to calculate reactions." << std::endl;
}
}
#endif

KRATOS_CATCH("");
}


void SetUpSystem(ModelPart& rModelPart) override
{
int free_id = 0;
int fix_id = BaseType::mDofSet.size();

for (auto dof_iterator = BaseType::mDofSet.begin(); dof_iterator != BaseType::mDofSet.end(); ++dof_iterator)
if (dof_iterator->IsFixed())
dof_iterator->SetEquationId(--fix_id);
else
dof_iterator->SetEquationId(free_id++);

BaseType::mEquationSystemSize = fix_id;

}


void ResizeAndInitializeVectors(
typename TSchemeType::Pointer pScheme,
TSystemMatrixPointerType& pA,
TSystemVectorPointerType& pDx,
TSystemVectorPointerType& pb,
ModelPart& rModelPart
) override
{
KRATOS_TRY

if (pA == nullptr) { 

TSystemMatrixPointerType pNewA = TSystemMatrixPointerType(new TSystemMatrixType(0, 0));
pA.swap(pNewA);
}
if (pDx == nullptr) { 

TSystemVectorPointerType pNewDx = TSystemVectorPointerType(new TSystemVectorType(0));
pDx.swap(pNewDx);
}
if (pb == nullptr) { 

TSystemVectorPointerType pNewb = TSystemVectorPointerType(new TSystemVectorType(0));
pb.swap(pNewb);
}
if (BaseType::mpReactionsVector == nullptr) { 

TSystemVectorPointerType pNewReactionsVector = TSystemVectorPointerType(new TSystemVectorType(0));
BaseType::mpReactionsVector.swap(pNewReactionsVector);
}

TSystemMatrixType& rA = *pA;
TSystemVectorType& rDx = *pDx;
TSystemVectorType& rb = *pb;

if (rA.size1() == 0 || BaseType::GetReshapeMatrixFlag()) { 
rA.resize(BaseType::mEquationSystemSize, BaseType::mEquationSystemSize, false);
ConstructMatrixStructure(pScheme, rA, rModelPart);
} else {
if (rA.size1() != BaseType::mEquationSystemSize || rA.size2() != BaseType::mEquationSystemSize) {
KRATOS_ERROR <<"The equation system size has changed during the simulation. This is not permitted."<<std::endl;
rA.resize(BaseType::mEquationSystemSize, BaseType::mEquationSystemSize, true);
ConstructMatrixStructure(pScheme, rA, rModelPart);
}
}
if (rDx.size() != BaseType::mEquationSystemSize) {
rDx.resize(BaseType::mEquationSystemSize, false);
}
TSparseSpace::SetToZero(rDx);
if (rb.size() != BaseType::mEquationSystemSize) {
rb.resize(BaseType::mEquationSystemSize, false);
}
TSparseSpace::SetToZero(rb);

if (BaseType::mCalculateReactionsFlag == true) {
const std::size_t reactions_vector_size = BaseType::mDofSet.size() - BaseType::mEquationSystemSize;
if (BaseType::mpReactionsVector->size() != reactions_vector_size)
BaseType::mpReactionsVector->resize(reactions_vector_size, false);
}

KRATOS_CATCH("")
}



void CalculateReactions(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemMatrixType& rA,
TSystemVectorType& rDx,
TSystemVectorType& rb
) override
{
BuildRHS(pScheme, rModelPart, rb);

std::size_t i;
TSystemVectorType& r_reactions_vector = *BaseType::mpReactionsVector;
for (auto it2 = BaseType::mDofSet.ptr_begin(); it2 != BaseType::mDofSet.ptr_end(); ++it2) {
i = (*it2)->EquationId();
if (i >= BaseType::mEquationSystemSize) {
i -= BaseType::mEquationSystemSize;
(*it2)->GetSolutionStepReactionValue() = -r_reactions_vector[i];
}

}
}


void ApplyDirichletConditions(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemMatrixType& rA,
TSystemVectorType& rDx,
TSystemVectorType& rb
) override
{
mScaleFactor = TSparseSpace::CheckAndCorrectZeroDiagonalValues(rModelPart.GetProcessInfo(), rA, rb, mScalingDiagonal); 
}


void Clear() override
{
this->mDofSet = DofsArrayType();
this->mpReactionsVector.reset();

this->mpLinearSystemSolver->Clear();

KRATOS_INFO_IF("ResidualBasedEliminationBuilderAndSolver", this->GetEchoLevel() > 1) << "Clear Function called" << std::endl;
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
"name"                                 : "elimination_builder_and_solver",
"block_builder"                        : false,
"diagonal_values_for_dirichlet_dofs"   : "use_max_diagonal"
})");

const Parameters base_default_parameters = BaseType::GetDefaultParameters();
default_parameters.RecursivelyAddMissingParameters(base_default_parameters);
return default_parameters;
}


static std::string Name()
{
return "elimination_builder_and_solver";
}




std::string Info() const override
{
return "ResidualBasedEliminationBuilderAndSolver";
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


#ifdef USE_LOCKS_IN_ASSEMBLY
std::vector<omp_lock_t> mLockArray; 
#endif

double mScaleFactor = 1.0;         

SCALING_DIAGONAL mScalingDiagonal = SCALING_DIAGONAL::NO_SCALING;; 





void Assemble(
TSystemMatrixType& rA,
TSystemVectorType& rb,
const LocalSystemMatrixType& rLHSContribution,
const LocalSystemVectorType& rRHSContribution,
const Element::EquationIdVectorType& rEquationId
#ifdef USE_LOCKS_IN_ASSEMBLY
,std::vector< omp_lock_t >& rLockArray
#endif
)
{
const SizeType local_size = rLHSContribution.size1();

for (IndexType i_local = 0; i_local < local_size; ++i_local) {
const IndexType i_global = rEquationId[i_local];

if (i_global < BaseType::mEquationSystemSize) {
#ifdef USE_LOCKS_IN_ASSEMBLY
omp_set_lock(&rLockArray[i_global]);
rb[i_global] += rRHSContribution(i_local);
#else
double& r_a = rb[i_global];
const double& v_a = rRHSContribution(i_local);
AtomicAdd(r_a, v_a);
#endif
AssembleRowContributionFreeDofs(rA, rLHSContribution, i_global, i_local, rEquationId);

#ifdef USE_LOCKS_IN_ASSEMBLY
omp_unset_lock(&rLockArray[i_global]);
#endif
}
}
}


virtual void ConstructMatrixStructure(
typename TSchemeType::Pointer pScheme,
TSystemMatrixType& rA,
ModelPart& rModelPart
)
{
Timer::Start("MatrixStructure");

const SizeType equation_size = BaseType::mEquationSystemSize;

std::vector<std::unordered_set<IndexType> > indices(equation_size);

block_for_each(indices, [](std::unordered_set<IndexType>& rIndices){
rIndices.reserve(40);
});

Element::EquationIdVectorType ids(3, 0);

#pragma omp parallel firstprivate(ids)
{
const ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();

std::vector<std::unordered_set<IndexType> > temp_indexes(equation_size);

#pragma omp for
for (int index = 0; index < static_cast<int>(equation_size); ++index)
temp_indexes[index].reserve(30);

const int number_of_elements = static_cast<int>(rModelPart.Elements().size());

const auto it_elem_begin = rModelPart.ElementsBegin();

#pragma omp for schedule(guided, 512) nowait
for (int i_elem = 0; i_elem<number_of_elements; ++i_elem) {
auto it_elem = it_elem_begin + i_elem;
pScheme->EquationId( *it_elem, ids, r_current_process_info);

for (auto& id_i : ids) {
if (id_i < BaseType::mEquationSystemSize) {
auto& row_indices = temp_indexes[id_i];
for (auto& id_j : ids)
if (id_j < BaseType::mEquationSystemSize)
row_indices.insert(id_j);
}
}
}

const int number_of_conditions = static_cast<int>(rModelPart.Conditions().size());

const auto it_cond_begin = rModelPart.ConditionsBegin();

#pragma omp for schedule(guided, 512) nowait
for (int i_cond = 0; i_cond<number_of_conditions; ++i_cond) {
auto it_cond = it_cond_begin + i_cond;
pScheme->EquationId( *it_cond, ids, r_current_process_info);

for (auto& id_i : ids) {
if (id_i < BaseType::mEquationSystemSize) {
auto& row_indices = temp_indexes[id_i];
for (auto& id_j : ids)
if (id_j < BaseType::mEquationSystemSize)
row_indices.insert(id_j);
}
}
}

#pragma omp critical
{
for (int i = 0; i < static_cast<int>(temp_indexes.size()); ++i) {
indices[i].insert(temp_indexes[i].begin(), temp_indexes[i].end());
}
}
}

SizeType nnz = 0;
for (IndexType i = 0; i < indices.size(); ++i)
nnz += indices[i].size();

rA = TSystemMatrixType(indices.size(), indices.size(), nnz);

double* Avalues = rA.value_data().begin();
std::size_t* Arow_indices = rA.index1_data().begin();
std::size_t* Acol_indices = rA.index2_data().begin();

Arow_indices[0] = 0;
for (IndexType i = 0; i < rA.size1(); ++i)
Arow_indices[i + 1] = Arow_indices[i] + indices[i].size();

IndexPartition<std::size_t>(rA.size1()).for_each([&](std::size_t Index){
const IndexType row_begin = Arow_indices[Index];
const IndexType row_end = Arow_indices[Index + 1];
IndexType k = row_begin;
for (auto it = indices[Index].begin(); it != indices[Index].end(); ++it) {
Acol_indices[k] = *it;
Avalues[k] = 0.0;
++k;
}

std::sort(&Acol_indices[row_begin], &Acol_indices[row_end]);
});

rA.set_filled(indices.size() + 1, nnz);

Timer::Stop("MatrixStructure");
}


void AssembleLHS(
TSystemMatrixType& rA,
LocalSystemMatrixType& rLHSContribution,
EquationIdVectorType& rEquationId
)
{
const SizeType local_size = rLHSContribution.size1();

for (IndexType i_local = 0; i_local < local_size; ++i_local) {
const IndexType i_global = rEquationId[i_local];
if (i_global < BaseType::mEquationSystemSize) {
for (IndexType j_local = 0; j_local < local_size; ++j_local) {
const IndexType j_global = rEquationId[j_local];
if (j_global < BaseType::mEquationSystemSize) {
rA(i_global, j_global) += rLHSContribution(i_local, j_local);
}
}
}
}
}


inline void AssembleRowContributionFreeDofs(
TSystemMatrixType& rA,
const Matrix& rALocal,
const IndexType i,
const IndexType i_local,
const Element::EquationIdVectorType& EquationId
)
{
double* values_vector = rA.value_data().begin();
IndexType* index1_vector = rA.index1_data().begin();
IndexType* index2_vector = rA.index2_data().begin();

const IndexType left_limit = index1_vector[i];

IndexType last_pos = 0;
IndexType last_found = 0;
IndexType counter = 0;
for(IndexType j=0; j < EquationId.size(); ++j) {
++counter;
const IndexType j_global = EquationId[j];
if (j_global < BaseType::mEquationSystemSize) {
last_pos = ForwardFind(j_global,left_limit,index2_vector);
last_found = j_global;
break;
}
}

if (counter <= EquationId.size()) {
#ifndef USE_LOCKS_IN_ASSEMBLY
double& r_a = values_vector[last_pos];
const double& v_a = rALocal(i_local,counter - 1);
AtomicAdd(r_a,  v_a);
#else
values_vector[last_pos] += rALocal(i_local,counter - 1);
#endif
IndexType pos = 0;
for(IndexType j = counter; j < EquationId.size(); ++j) {
IndexType id_to_find = EquationId[j];
if (id_to_find < BaseType::mEquationSystemSize) {
if(id_to_find > last_found)
pos = ForwardFind(id_to_find,last_pos+1,index2_vector);
else if(id_to_find < last_found)
pos = BackwardFind(id_to_find,last_pos-1,index2_vector);
else
pos = last_pos;

#ifndef USE_LOCKS_IN_ASSEMBLY
double& r = values_vector[pos];
const double& v = rALocal(i_local,j);
AtomicAdd(r,  v);
#else
values_vector[pos] += rALocal(i_local,j);
#endif
last_found = id_to_find;
last_pos = pos;
}
}
}
}

inline IndexType ForwardFind(const IndexType id_to_find,
const IndexType start,
const IndexType* index_vector)
{
IndexType pos = start;
while(id_to_find != index_vector[pos]) pos++;
return pos;
}

inline IndexType BackwardFind(const IndexType id_to_find,
const IndexType start,
const IndexType* index_vector)
{
IndexType pos = start;
while(id_to_find != index_vector[pos]) pos--;
return pos;
}


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


void AssembleRHS(
TSystemVectorType& rb,
const LocalSystemVectorType& rRHSContribution,
const EquationIdVectorType& rEquationId
)
{
SizeType local_size = rRHSContribution.size();

if (BaseType::mCalculateReactionsFlag == false) {
for (IndexType i_local = 0; i_local < local_size; ++i_local) {
const IndexType i_global = rEquationId[i_local];

if (i_global < BaseType::mEquationSystemSize) { 
double& b_value = rb[i_global];
const double& rhs_value = rRHSContribution[i_local];

AtomicAdd(b_value, rhs_value);
}
}
} else {
TSystemVectorType& r_reactions_vector = *BaseType::mpReactionsVector;
for (IndexType i_local = 0; i_local < local_size; ++i_local) {
const IndexType i_global = rEquationId[i_local];

if (i_global < BaseType::mEquationSystemSize) { 
double& b_value = rb[i_global];
const double& rhs_value = rRHSContribution[i_local];

AtomicAdd(b_value, rhs_value);
} else { 
double& b_value = r_reactions_vector[i_global - BaseType::mEquationSystemSize];
const double& rhs_value = rRHSContribution[i_local];

AtomicAdd(b_value, rhs_value);
}
}
}
}


void AssembleLHSCompleteOnFreeRows(
TSystemMatrixType& rA,
LocalSystemMatrixType& rLHSContribution,
EquationIdVectorType& rEquationId
)
{
const SizeType local_size = rLHSContribution.size1();
for (IndexType i_local = 0; i_local < local_size; ++i_local) {
const IndexType i_global = rEquationId[i_local];
if (i_global < BaseType::mEquationSystemSize) {
for (IndexType j_local = 0; j_local < local_size; ++j_local) {
const IndexType j_global = rEquationId[j_local];
rA(i_global, j_global) += rLHSContribution(i_local, j_local);
}
}
}
}






}; 




} 

#endif 
