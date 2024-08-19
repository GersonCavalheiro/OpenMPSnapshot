
#if !defined(KRATOS_REDUCTION_BUILDER_AND_SOLVER_H_INCLUDED)
#define  KRATOS_REDUCTION_BUILDER_AND_SOLVER_H_INCLUDED



#include "custom_solvers/solution_builders_and_solvers/solution_builder_and_solver.hpp"
#include "includes/key_hash.h"

#ifdef USE_GOOGLE_HASH
#include "sparsehash/dense_hash_set" 
#else
#include <unordered_set>
#endif


namespace Kratos
{






template<class TSparseSpace,
class TDenseSpace, 
class TLinearSolver 
>
class ReductionBuilderAndSolver : public SolutionBuilderAndSolver< TSparseSpace, TDenseSpace, TLinearSolver >
{
public:

KRATOS_CLASS_POINTER_DEFINITION(ReductionBuilderAndSolver);

typedef SolutionBuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver>      BaseType;
typedef typename BaseType::Pointer                                       BasePointerType;

typedef typename BaseType::LocalFlagType                                   LocalFlagType;
typedef typename BaseType::DofsArrayType                                   DofsArrayType;

typedef typename BaseType::SystemMatrixType                             SystemMatrixType;
typedef typename BaseType::SystemVectorType                             SystemVectorType;
typedef typename BaseType::SystemMatrixPointerType               SystemMatrixPointerType;
typedef typename BaseType::SystemVectorPointerType               SystemVectorPointerType;
typedef typename BaseType::LocalSystemVectorType                   LocalSystemVectorType;
typedef typename BaseType::LocalSystemMatrixType                   LocalSystemMatrixType;

typedef typename ModelPart::NodesContainerType                        NodesContainerType;
typedef typename ModelPart::ElementsContainerType                  ElementsContainerType;
typedef typename ModelPart::ConditionsContainerType              ConditionsContainerType;

typedef typename BaseType::SchemePointerType                           SchemePointerType;
typedef typename BaseType::LinearSolverPointerType               LinearSolverPointerType;

struct dof_iterator_hash
{
size_t operator()(const Node::DofType::Pointer& it) const
{
std::size_t seed = 0;
HashCombine(seed, it->Id());
HashCombine(seed, (it->GetVariable()).Key());
return seed;
}
};

struct dof_iterator_equal
{
size_t operator()(const Node::DofType::Pointer& it1, const Node::DofType::Pointer& it2) const
{
return (((it1->Id() == it2->Id() && (it1->GetVariable()).Key()) == (it2->GetVariable()).Key()));
}
};


ReductionBuilderAndSolver(LinearSolverPointerType pLinearSystemSolver)
: BaseType(pLinearSystemSolver)
{
}

~ReductionBuilderAndSolver() override
{
}




void BuildLHS(SchemePointerType pScheme,
ModelPart& rModelPart,
SystemMatrixType& rA) override
{
KRATOS_TRY

ElementsContainerType& rElements = rModelPart.Elements();

ConditionsContainerType& rConditions = rModelPart.Conditions();

TSparseSpace::SetToZero(*(this->mpReactionsVector));

LocalSystemMatrixType LHS_Contribution = LocalSystemMatrixType(0, 0);

Element::EquationIdVectorType EquationId;

ProcessInfo& rCurrentProcessInfo = rModelPart.GetProcessInfo();

for (typename ElementsContainerType::ptr_iterator it = rElements.ptr_begin(); it != rElements.ptr_end(); ++it)
{
pScheme->Calculate_LHS_Contribution(*it, LHS_Contribution, EquationId, rCurrentProcessInfo);

AssembleLHS(rA, LHS_Contribution, EquationId);

pScheme->Clear(*it);
}

LHS_Contribution.resize(0, 0, false);

for (typename ConditionsContainerType::ptr_iterator it = rConditions.ptr_begin(); it != rConditions.ptr_end(); ++it)
{
pScheme->Condition_Calculate_LHS_Contribution(*it, LHS_Contribution, EquationId, rCurrentProcessInfo);

AssembleLHS(rA, LHS_Contribution, EquationId);
}

KRATOS_CATCH("")

}


void BuildRHS(SchemePointerType pScheme,
ModelPart& rModelPart,
SystemVectorType& rb) override
{
KRATOS_TRY


if(this->mOptions.Is(LocalFlagType::COMPUTE_REACTIONS))
{
TSparseSpace::SetToZero(*(this->mpReactionsVector));
}

ElementsContainerType& rElements = rModelPart.Elements();

ConditionsContainerType& rConditions = rModelPart.Conditions();

ProcessInfo& rCurrentProcessInfo = rModelPart.GetProcessInfo();

LocalSystemVectorType RHS_Contribution = LocalSystemVectorType(0);

Element::EquationIdVectorType EquationId;


#pragma omp parallel firstprivate( RHS_Contribution, EquationId)
{
const int nelements = static_cast<int>(rElements.size());
#pragma omp for schedule(guided, 512) nowait
for (int i = 0; i<nelements; i++)
{
typename ElementsContainerType::iterator it = rElements.begin() + i;
bool element_is_active = true;
if ((it)->IsDefined(ACTIVE))
element_is_active = (it)->Is(ACTIVE);

if (element_is_active)
{
pScheme->Calculate_RHS_Contribution(*(it.base()), RHS_Contribution, EquationId, rCurrentProcessInfo);

AssembleRHS(rb, RHS_Contribution, EquationId);
}
}

const int nconditions = static_cast<int>(rConditions.size());
#pragma omp for schedule(guided, 512)
for (int i = 0; i<nconditions; i++)
{
auto it = rConditions.begin() + i;
bool condition_is_active = true;
if ((it)->IsDefined(ACTIVE))
condition_is_active = (it)->Is(ACTIVE);

if (condition_is_active)
{
pScheme->Condition_Calculate_RHS_Contribution(*(it.base()), RHS_Contribution, EquationId, rCurrentProcessInfo);

AssembleRHS(rb, RHS_Contribution, EquationId);
}
}
}


KRATOS_CATCH("")
}


void Build(SchemePointerType pScheme,
ModelPart& rModelPart,
SystemMatrixType& rA,
SystemVectorType& rb) override
{
KRATOS_TRY

if (!pScheme)
KRATOS_ERROR << "No scheme provided!" << std::endl;

const int nelements = static_cast<int>(rModelPart.Elements().size());

const int nconditions = static_cast<int>(rModelPart.Conditions().size());

ProcessInfo& rCurrentProcessInfo = rModelPart.GetProcessInfo();
ModelPart::ElementsContainerType::iterator el_begin = rModelPart.ElementsBegin();
ModelPart::ConditionsContainerType::iterator cond_begin = rModelPart.ConditionsBegin();

LocalSystemMatrixType LHS_Contribution = LocalSystemMatrixType(0, 0);
LocalSystemVectorType RHS_Contribution = LocalSystemVectorType(0);

Element::EquationIdVectorType EquationId;


#pragma omp parallel firstprivate(nelements, nconditions,  LHS_Contribution, RHS_Contribution, EquationId )
{
#pragma omp for schedule(guided, 512) nowait
for (int k = 0; k < nelements; k++)
{
ModelPart::ElementsContainerType::iterator it = el_begin + k;

bool element_is_active = true;
if ((it)->IsDefined(ACTIVE))
element_is_active = (it)->Is(ACTIVE);

if (element_is_active)
{
pScheme->CalculateSystemContributions(*(it.base()), LHS_Contribution, RHS_Contribution, EquationId, rCurrentProcessInfo);

#ifdef _OPENMP
Assemble(rA, rb, LHS_Contribution, RHS_Contribution, EquationId, mlock_array);
#else
Assemble(rA, rb, LHS_Contribution, RHS_Contribution, EquationId);
#endif
pScheme->Clear(*(it.base()));

}

}

#pragma omp for schedule(guided, 512)
for (int k = 0; k < nconditions; k++)
{
ModelPart::ConditionsContainerType::iterator it = cond_begin + k;

bool condition_is_active = true;
if ((it)->IsDefined(ACTIVE))
condition_is_active = (it)->Is(ACTIVE);

if (condition_is_active)
{
pScheme->Condition_CalculateSystemContributions(*(it.base()), LHS_Contribution, RHS_Contribution, EquationId, rCurrentProcessInfo);

#ifdef _OPENMP
Assemble(rA, rb, LHS_Contribution, RHS_Contribution, EquationId, mlock_array);
#else
Assemble(rA, rb, LHS_Contribution, RHS_Contribution, EquationId);
#endif

pScheme->Clear(*(it.base()));
}
}
}


if (this->mEchoLevel > 2 && rModelPart.GetCommunicator().MyPID() == 0){
KRATOS_INFO("parallel_build") << "finished" << std::endl;
}

KRATOS_CATCH("")

}


void SystemSolve(SystemMatrixType& rA,
SystemVectorType& rDx,
SystemVectorType& rb) override
{
KRATOS_TRY

double norm_b;
if (TSparseSpace::Size(rb) != 0)
norm_b = TSparseSpace::TwoNorm(rb);
else
norm_b = 0.00;

if (norm_b != 0.00)
{
this->mpLinearSystemSolver->Solve(rA, rDx, rb);
}
else
TSparseSpace::SetToZero(rDx);

if (this->GetEchoLevel() > 1)
{
KRATOS_INFO("linear_solver") << *(this->mpLinearSystemSolver) << std::endl;
}

KRATOS_CATCH("")

}


void BuildAndSolve(SchemePointerType pScheme,
ModelPart& rModelPart,
SystemMatrixType& rA,
SystemVectorType& rDx,
SystemVectorType& rb) override
{
KRATOS_TRY

Build(pScheme, rModelPart, rA, rb);


ApplyDirichletConditions(pScheme, rModelPart, rA, rDx, rb);

if (this->mEchoLevel == 3)
{
KRATOS_INFO("LHS before solve") << "Matrix = " << rA << std::endl;
KRATOS_INFO("Dx before solve")  << "Solution = " << rDx << std::endl;
KRATOS_INFO("RHS before solve") << "Vector = " << rb << std::endl;
}

SystemSolveWithPhysics(rA, rDx, rb, rModelPart);



if (this->mEchoLevel == 3)
{
KRATOS_INFO("LHS after solve") << "Matrix = " << rA << std::endl;
KRATOS_INFO("Dx after solve")  << "Solution = " << rDx << std::endl;
KRATOS_INFO("RHS after solve") << "Vector = " << rb << std::endl;
}

KRATOS_CATCH("")
}


void BuildRHSAndSolve(SchemePointerType pScheme,
ModelPart& rModelPart,
SystemMatrixType& rA,
SystemVectorType& rDx,
SystemVectorType& rb) override
{
KRATOS_TRY

BuildRHS(pScheme, rModelPart, rb);
SystemSolve(rA, rDx, rb);

KRATOS_CATCH("")
}



void ApplyDirichletConditions(SchemePointerType pScheme,
ModelPart& rModelPart,
SystemMatrixType& rA,
SystemVectorType& rDx,
SystemVectorType& rb) override
{
}


void SetUpDofSet(SchemePointerType pScheme,
ModelPart& rModelPart) override
{
KRATOS_TRY

if( this->mEchoLevel > 1 && rModelPart.GetCommunicator().MyPID() == 0)
{
KRATOS_INFO("setting_dofs") << "SetUpDofSet starts" << std::endl;
}

ElementsContainerType& rElements = rModelPart.Elements();
const int nelements = static_cast<int>(rElements.size());

Element::DofsVectorType ElementalDofList;

ProcessInfo& rCurrentProcessInfo = rModelPart.GetProcessInfo();

unsigned int nthreads = ParallelUtilities::GetNumThreads();

#ifdef USE_GOOGLE_HASH
typedef google::dense_hash_set < Node::DofType::Pointer, dof_iterator_hash>  set_type;
#else
typedef std::unordered_set < Node::DofType::Pointer, dof_iterator_hash>  set_type;
#endif

std::vector<set_type> dofs_aux_list(nthreads);

if( this->mEchoLevel > 2)
{
KRATOS_INFO("setting_dofs") << "Number of threads:" << nthreads << std::endl;
}

for (int i = 0; i < static_cast<int>(nthreads); i++)
{
#ifdef USE_GOOGLE_HASH
dofs_aux_list[i].set_empty_key(Node::DofType::Pointer());
#else
dofs_aux_list[i].reserve(nelements);
#endif
}

if( this->mEchoLevel > 2)
{
KRATOS_INFO("setting_dofs") << "initialize_elements" << std::endl;
}

#pragma omp parallel for firstprivate(nelements, ElementalDofList)
for (int i = 0; i < static_cast<int>(nelements); i++)
{
typename ElementsContainerType::iterator it = rElements.begin() + i;
const unsigned int this_thread_id = OpenMPUtils::ThisThread();

pScheme->GetElementalDofList(*(it.base()), ElementalDofList, rCurrentProcessInfo);

dofs_aux_list[this_thread_id].insert(ElementalDofList.begin(), ElementalDofList.end());
}

if( this->mEchoLevel > 2)
{
KRATOS_INFO("setting_dofs") << "initialize_conditions" << std::endl;
}

ConditionsContainerType& rConditions = rModelPart.Conditions();
const int nconditions = static_cast<int>(rConditions.size());
#pragma omp parallel for firstprivate(nconditions, ElementalDofList)
for (int i = 0; i < nconditions; i++)
{
typename ConditionsContainerType::iterator it = rConditions.begin() + i;
const unsigned int this_thread_id = OpenMPUtils::ThisThread();

pScheme->GetConditionDofList(*(it.base()), ElementalDofList, rCurrentProcessInfo);
dofs_aux_list[this_thread_id].insert(ElementalDofList.begin(), ElementalDofList.end());

}

if( this->mEchoLevel > 2)
{
KRATOS_INFO("setting_dofs") << "initialize tree reduction" << std::endl;
}

unsigned int old_max = nthreads;
unsigned int new_max = ceil(0.5*static_cast<double>(old_max));
while (new_max >= 1 && new_max != old_max)
{
if( this->mEchoLevel > 2)
{
KRATOS_INFO("setting_dofs") << "old_max" << old_max << " new_max:" << new_max << std::endl;
for (int i = 0; i < static_cast<int>(new_max); i++)
{
if (i + new_max < old_max)
{
KRATOS_INFO("setting_dofs") << i << " - " << i+new_max << std::endl;
}
}
}

#pragma omp parallel for
for (int i = 0; i < static_cast<int>(new_max); i++)
{
if (i + new_max < old_max)
{
dofs_aux_list[i].insert(dofs_aux_list[i + new_max].begin(), dofs_aux_list[i + new_max].end());
dofs_aux_list[i + new_max].clear();
}
}

old_max = new_max;
new_max = ceil(0.5*static_cast<double>(old_max));

}

if( this->mEchoLevel > 2)
{
KRATOS_INFO("setting_dofs") << "initializing ordered array filling" << std::endl;
}

DofsArrayType Doftemp;
this->mDofSet = DofsArrayType();

Doftemp.reserve(dofs_aux_list[0].size());
for (auto it = dofs_aux_list[0].begin(); it != dofs_aux_list[0].end(); it++)
{
Doftemp.push_back(*it);
}
Doftemp.Sort();

this->mDofSet = Doftemp;

if (this->mDofSet.size() == 0)
{
KRATOS_ERROR << "No degrees of freedom!" << std::endl;
}
if( this->mEchoLevel > 2)
{
KRATOS_INFO("Dofs size") << this->mDofSet.size() << std::endl;
}

if( this->mEchoLevel > 2 && rModelPart.GetCommunicator().MyPID() == 0)
{
KRATOS_INFO("setting_dofs") << "Finished setting up the dofs" << std::endl;
}

if( this->mEchoLevel > 2)
{
KRATOS_INFO("setting_dofs") << "Initializing lock array" << std::endl;
}

#ifdef _OPENMP
if (mlock_array.size() != 0)
{
for (int i = 0; i < static_cast<int>(mlock_array.size()); i++)
omp_destroy_lock(&mlock_array[i]);
}

mlock_array.resize(this->mDofSet.size());

for (int i = 0; i < static_cast<int>(mlock_array.size()); i++)
omp_init_lock(&mlock_array[i]);
#endif

if( this->mEchoLevel > 2)
{
KRATOS_INFO("setting_dofs") << "End of setupdofset" << std::endl;
}

this->Set(LocalFlagType::DOFS_INITIALIZED, true);

KRATOS_CATCH("");
}


void SetUpSystem() override
{
int free_id = 0;
int fix_id = this->mDofSet.size();

for (typename DofsArrayType::iterator dof_iterator = this->mDofSet.begin(); dof_iterator != this->mDofSet.end(); ++dof_iterator)
if (dof_iterator->IsFixed())
dof_iterator->SetEquationId(--fix_id);
else
dof_iterator->SetEquationId(free_id++);

this->mEquationSystemSize = fix_id;

}


void SetUpSystemMatrices(SchemePointerType pScheme,
ModelPart& rModelPart,
SystemMatrixPointerType& pA,
SystemVectorPointerType& pDx,
SystemVectorPointerType& pb) override
{
KRATOS_TRY

if (pA == nullptr) 
{
SystemMatrixPointerType pNewA = Kratos::make_shared<SystemMatrixType>(0, 0);
pA.swap(pNewA);
}
if (pDx == nullptr) 
{
SystemVectorPointerType pNewDx = Kratos::make_shared<SystemVectorType>(0);
pDx.swap(pNewDx);
}
if (pb == nullptr) 
{
SystemVectorPointerType pNewb = Kratos::make_shared<SystemVectorType>(0);
pb.swap(pNewb);
}
if (this->mpReactionsVector == nullptr) 
{
SystemVectorPointerType pNewReactionsVector = Kratos::make_shared<SystemVectorType>(0);
this->mpReactionsVector.swap(pNewReactionsVector);
}

SystemMatrixType& rA = *pA;
SystemVectorType& rDx = *pDx;
SystemVectorType& rb = *pb;

if (rA.size1() == 0 || this->mOptions.Is(LocalFlagType::REFORM_DOFS)) 
{
rA.resize(this->mEquationSystemSize, this->mEquationSystemSize, false);
ConstructMatrixStructure(pScheme, rA, rModelPart.Elements(), rModelPart.Conditions(), rModelPart.GetProcessInfo());
}
else
{
if (rA.size1() != this->mEquationSystemSize || rA.size2() != this->mEquationSystemSize)
{
KRATOS_WARNING("reduction builder resize") << "it should not come here -> this is SLOW" << std::endl;
rA.resize(this->mEquationSystemSize, this->mEquationSystemSize, true);
ConstructMatrixStructure(pScheme, rA, rModelPart.Elements(), rModelPart.Conditions(), rModelPart.GetProcessInfo());
}
}
if (rDx.size() != this->mEquationSystemSize)
rDx.resize(this->mEquationSystemSize, false);
if (rb.size() != this->mEquationSystemSize)
rb.resize(this->mEquationSystemSize, false);

if(this->mOptions.Is(LocalFlagType::COMPUTE_REACTIONS))
{
unsigned int ReactionsVectorSize = this->mDofSet.size();
if (this->mpReactionsVector->size() != ReactionsVectorSize)
this->mpReactionsVector->resize(ReactionsVectorSize, false);
}

KRATOS_CATCH("")

}


void InitializeSolutionStep(SchemePointerType pScheme,
ModelPart& rModelPart,
SystemMatrixPointerType& pA,
SystemVectorPointerType& pDx,
SystemVectorPointerType& pb) override
{
KRATOS_TRY

BaseType::InitializeSolutionStep(pScheme, rModelPart, pA, pDx, pb);





KRATOS_CATCH("")
}


void FinalizeSolutionStep(SchemePointerType pScheme,
ModelPart& rModelPart,
SystemMatrixPointerType& pA,
SystemVectorPointerType& pDx,
SystemVectorPointerType& pb) override
{
KRATOS_TRY

BaseType::FinalizeSolutionStep(pScheme, rModelPart, pA, pDx, pb);

KRATOS_CATCH("")
}


void CalculateReactions(SchemePointerType pScheme,
ModelPart& rModelPart,
SystemMatrixType& rA,
SystemVectorType& rDx,
SystemVectorType& rb) override
{
BuildRHS(pScheme, rModelPart, rb);

int i;
int systemsize = this->mDofSet.size() - TSparseSpace::Size(*this->mpReactionsVector);

typename DofsArrayType::ptr_iterator it2;

SystemVectorType& ReactionsVector = *this->mpReactionsVector;
for (it2 = this->mDofSet.ptr_begin(); it2 != this->mDofSet.ptr_end(); ++it2)
{
i = (*it2)->EquationId();
i -= systemsize;
(*it2)->GetSolutionStepReactionValue() = -ReactionsVector[i];

}
}



void Clear() override
{

BaseType::Clear();

#ifdef _OPENMP
for (int i = 0; i < static_cast<int>(mlock_array.size()); i++)
omp_destroy_lock(&mlock_array[i]);
mlock_array.resize(0);
#endif

}


int Check(ModelPart& rModelPart) override
{
KRATOS_TRY

return 0;

KRATOS_CATCH("");
}





protected:


#ifdef _OPENMP
std::vector< omp_lock_t > mlock_array;
#endif





void SystemSolveWithPhysics(SystemMatrixType& rA,
SystemVectorType& rDx,
SystemVectorType& rb,
ModelPart& rModelPart)
{
KRATOS_TRY

double norm_b;
if (TSparseSpace::Size(rb) != 0)
norm_b = TSparseSpace::TwoNorm(rb);
else
norm_b = 0.00;

if (norm_b != 0.00)
{
if(this->mpLinearSystemSolver->AdditionalPhysicalDataIsNeeded() )
this->mpLinearSystemSolver->ProvideAdditionalData(rA, rDx, rb, this->mDofSet, rModelPart);

this->mpLinearSystemSolver->Solve(rA, rDx, rb);
}
else
{
TSparseSpace::SetToZero(rDx);
KRATOS_WARNING("RHS") << "ATTENTION! setting the RHS to zero!" << std::endl;
}

if (this->GetEchoLevel() > 1)
{
KRATOS_INFO("LinearSolver") << *(this->mpLinearSystemSolver) << std::endl;
}

KRATOS_CATCH("")

}

virtual void ConstructMatrixStructure(SchemePointerType pScheme,
SystemMatrixType& rA,
ElementsContainerType& rElements,
ConditionsContainerType& rConditions,
ProcessInfo& rCurrentProcessInfo)
{

const std::size_t equation_size = this->mEquationSystemSize;

#ifdef USE_GOOGLE_HASH
std::vector<google::dense_hash_set<std::size_t> > indices(equation_size);
const std::size_t empty_key = 2 * equation_size + 10;
#else
std::vector<std::unordered_set<std::size_t> > indices(equation_size);
#endif

#pragma omp parallel for firstprivate(equation_size)
for (int iii = 0; iii < static_cast<int>(equation_size); iii++)
{
#ifdef USE_GOOGLE_HASH
indices[iii].set_empty_key(empty_key);
#else
indices[iii].reserve(40);
#endif
}

Element::EquationIdVectorType ids(3, 0);

const int nelements = static_cast<int>(rElements.size());
#pragma omp parallel for firstprivate(nelements, ids)
for (int iii = 0; iii<nelements; iii++)
{
typename ElementsContainerType::iterator i_element = rElements.begin() + iii;
pScheme->EquationId( *(i_element.base()), ids, rCurrentProcessInfo);

for (std::size_t i = 0; i < ids.size(); i++)
{
if (ids[i] < this->mEquationSystemSize)
{
#ifdef _OPENMP
omp_set_lock(&mlock_array[ids[i]]);
#endif
auto& row_indices = indices[ids[i]];
for (auto it = ids.begin(); it != ids.end(); it++)
{
if (*it < this->mEquationSystemSize)
row_indices.insert(*it);
}
#ifdef _OPENMP
omp_unset_lock(&mlock_array[ids[i]]);
#endif
}
}

}

const int nconditions = static_cast<int>(rConditions.size());
#pragma omp parallel for firstprivate(nconditions, ids)
for (int iii = 0; iii<nconditions; iii++)
{
typename ConditionsContainerType::iterator i_condition = rConditions.begin() + iii;
pScheme->Condition_EquationId( *(i_condition.base()) , ids, rCurrentProcessInfo);
for (std::size_t i = 0; i < ids.size(); i++)
{
if (ids[i] < this->mEquationSystemSize)
{
#ifdef _OPENMP
omp_set_lock(&mlock_array[ids[i]]);
#endif
auto& row_indices = indices[ids[i]];
for (auto it = ids.begin(); it != ids.end(); it++)
{
if (*it < this->mEquationSystemSize)
row_indices.insert(*it);
}
#ifdef _OPENMP
omp_unset_lock(&mlock_array[ids[i]]);
#endif
}
}
}

unsigned int nnz = 0;
for (unsigned int i = 0; i < indices.size(); i++)
nnz += indices[i].size();

rA = boost::numeric::ublas::compressed_matrix<double>(indices.size(), indices.size(), nnz);

double* Avalues = rA.value_data().begin();
std::size_t* Arow_indices = rA.index1_data().begin();
std::size_t* Acol_indices = rA.index2_data().begin();

Arow_indices[0] = 0;
for (int i = 0; i < static_cast<int>(rA.size1()); i++)
Arow_indices[i + 1] = Arow_indices[i] + indices[i].size();


#pragma omp parallel for
for (int i = 0; i < static_cast<int>(rA.size1()); i++)
{
const unsigned int row_begin = Arow_indices[i];
const unsigned int row_end = Arow_indices[i + 1];
unsigned int k = row_begin;
for (auto it = indices[i].begin(); it != indices[i].end(); it++)
{
Acol_indices[k] = *it;
Avalues[k] = 0.0;
k++;
}

indices[i].clear(); 

std::sort(&Acol_indices[row_begin], &Acol_indices[row_end]);

}

rA.set_filled(indices.size() + 1, nnz);


}

void AssembleLHS(SystemMatrixType& rA,
LocalSystemMatrixType& rLHS_Contribution,
Element::EquationIdVectorType& rEquationId)
{
unsigned int local_size = rLHS_Contribution.size1();

for (unsigned int i_local = 0; i_local < local_size; i_local++)
{
unsigned int i_global = rEquationId[i_local];
if (i_global < this->mEquationSystemSize)
{
for (unsigned int j_local = 0; j_local < local_size; j_local++)
{
unsigned int j_global = rEquationId[j_local];
if (j_global < this->mEquationSystemSize)
rA(i_global, j_global) += rLHS_Contribution(i_local, j_local);
}
}
}
}


void AssembleRHS(SystemVectorType& rb,
const LocalSystemVectorType& rRHS_Contribution,
const Element::EquationIdVectorType& rEquationId)
{
unsigned int local_size = rRHS_Contribution.size();

if(this->mOptions.IsNot(LocalFlagType::COMPUTE_REACTIONS))
{
for (unsigned int i_local = 0; i_local < local_size; i_local++)
{
const unsigned int i_global = rEquationId[i_local];

if (i_global < this->mEquationSystemSize) 
{
double& b_value = rb[i_global];
const double& rhs_value = rRHS_Contribution[i_local];

#pragma omp atomic
b_value += rhs_value;
}
}
}
else
{
SystemVectorType& ReactionsVector = *this->mpReactionsVector;
for (unsigned int i_local = 0; i_local < local_size; i_local++)
{
const unsigned int i_global = rEquationId[i_local];

if (i_global < this->mEquationSystemSize) 
{
double& b_value = rb[i_global];
const double& rhs_value = rRHS_Contribution[i_local];

#pragma omp atomic
b_value += rhs_value;
}
else 
{
double& b_value = ReactionsVector[i_global - this->mEquationSystemSize];
const double& rhs_value = rRHS_Contribution[i_local];

#pragma omp atomic
b_value += rhs_value;
}
}
}
}

void Assemble(SystemMatrixType& rA,
SystemVectorType& rb,
const LocalSystemMatrixType& rLHS_Contribution,
const LocalSystemVectorType& rRHS_Contribution,
const Element::EquationIdVectorType& rEquationId
#ifdef _OPENMP
,std::vector< omp_lock_t >& lock_array
#endif
)
{
unsigned int local_size = rLHS_Contribution.size1();

for (unsigned int i_local = 0; i_local < local_size; i_local++)
{
unsigned int i_global = rEquationId[i_local];

if (i_global < this->mEquationSystemSize)
{
#ifdef _OPENMP
omp_set_lock(&lock_array[i_global]);
#endif
rb[i_global] += rRHS_Contribution(i_local);
for (unsigned int j_local = 0; j_local < local_size; j_local++)
{
unsigned int j_global = rEquationId[j_local];
if (j_global < this->mEquationSystemSize)
{
rA(i_global, j_global) += rLHS_Contribution(i_local, j_local);
}
}
#ifdef _OPENMP
omp_unset_lock(&lock_array[i_global]);
#endif

}
}
}





private:





inline void AddUnique(std::vector<std::size_t>& v, const std::size_t& candidate)
{
std::vector<std::size_t>::iterator i = v.begin();
std::vector<std::size_t>::iterator endit = v.end();
while (i != endit && (*i) != candidate)
{
++i;
}
if (i == endit)
{
v.push_back(candidate);
}

}






}; 






}  

#endif 
