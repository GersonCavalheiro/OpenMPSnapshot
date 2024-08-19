
#if !defined(KRATOS_BLOCK_BUILDER_AND_SOLVER_H_INCLUDED)
#define  KRATOS_BLOCK_BUILDER_AND_SOLVER_H_INCLUDED



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
class BlockBuilderAndSolver : public SolutionBuilderAndSolver< TSparseSpace, TDenseSpace, TLinearSolver >
{
public:


KRATOS_CLASS_POINTER_DEFINITION(BlockBuilderAndSolver);

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
return (((it1->Id() == it2->Id() && (it1->GetVariable()).Key())==(it2->GetVariable()).Key()));
}
};



BlockBuilderAndSolver(LinearSolverPointerType pLinearSystemSolver)
: BaseType(pLinearSystemSolver)
{
}

~BlockBuilderAndSolver() override
{
}




void BuildLHS(SchemePointerType pScheme,
ModelPart& rModelPart,
SystemMatrixType& rA) override
{
KRATOS_TRY

SystemVectorType tmp(rA.size1(), 0.0);
this->Build(pScheme, rModelPart, rA, tmp);

KRATOS_CATCH("")
}


void BuildRHS(SchemePointerType pScheme,
ModelPart& rModelPart,
SystemVectorType& rb) override
{
KRATOS_TRY

BuildRHSNoDirichlet(pScheme,rModelPart,rb);

const int ndofs = static_cast<int>(this->mDofSet.size());

#pragma omp parallel for firstprivate(ndofs)
for (int k = 0; k<ndofs; k++)
{
typename DofsArrayType::iterator dof_iterator = this->mDofSet.begin() + k;
const std::size_t i = dof_iterator->EquationId();

if (dof_iterator->IsFixed())
rb[i] = 0.0f;
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


#pragma omp parallel firstprivate(nelements,nconditions, LHS_Contribution, RHS_Contribution, EquationId )
{
# pragma omp for  schedule(guided, 512) nowait
for (int k = 0; k < nelements; k++)
{
ModelPart::ElementsContainerType::iterator it = el_begin + k;

bool element_is_active = true;
if ((it)->IsDefined(ACTIVE))
element_is_active = (it)->Is(ACTIVE);

if (element_is_active)
{
pScheme->CalculateSystemContributions(*(it.base()), LHS_Contribution, RHS_Contribution, EquationId, rCurrentProcessInfo);

#ifdef USE_LOCKS_IN_ASSEMBLY
Assemble(rA, rb, LHS_Contribution, RHS_Contribution, EquationId, mlock_array);
#else
Assemble(rA, rb, LHS_Contribution, RHS_Contribution, EquationId);
#endif
pScheme->Clear(*(it.base()));
}

}


#pragma omp for  schedule(guided, 512)
for (int k = 0; k < nconditions; k++)
{
ModelPart::ConditionsContainerType::iterator it = cond_begin + k;

bool condition_is_active = true;
if ((it)->IsDefined(ACTIVE))
condition_is_active = (it)->Is(ACTIVE);

if (condition_is_active)
{
pScheme->Condition_CalculateSystemContributions(*(it.base()), LHS_Contribution, RHS_Contribution, EquationId, rCurrentProcessInfo);

#ifdef USE_LOCKS_IN_ASSEMBLY
Assemble(rA, rb, LHS_Contribution, RHS_Contribution, EquationId, mlock_array);
#else
Assemble(rA, rb, LHS_Contribution, RHS_Contribution, EquationId);
#endif

pScheme->Clear(*(it.base()));
}
}
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

if (this->mEchoLevel > 1)
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
std::size_t system_size = rA.size1();
std::vector<double> scaling_factors (system_size, 0.0f);

const int ndofs = static_cast<int>(this->mDofSet.size());

#pragma omp parallel for firstprivate(ndofs)
for(int k = 0; k<ndofs; k++)
{
typename DofsArrayType::iterator dof_iterator = this->mDofSet.begin() + k;
if(dof_iterator->IsFixed())
scaling_factors[k] = 0.0f;
else
scaling_factors[k] = 1.0f;

}

double* Avalues = rA.value_data().begin();
std::size_t* Arow_indices = rA.index1_data().begin();
std::size_t* Acol_indices = rA.index2_data().begin();

#pragma omp parallel for firstprivate(system_size)
for(int k = 0; k < static_cast<int>(system_size); ++k)
{
std::size_t col_begin = Arow_indices[k];
std::size_t col_end = Arow_indices[k+1];
bool empty = true;
for (std::size_t j = col_begin; j < col_end; ++j)
{
if(Avalues[j] != 0.0)
{
empty = false;
break;
}
}

if(empty == true)
{
rA(k,k) = 1.0;
rb[k] = 0.0;
}
}

#pragma omp parallel for
for (int k = 0; k < static_cast<int>(system_size); ++k)
{
std::size_t col_begin = Arow_indices[k];
std::size_t col_end = Arow_indices[k+1];
double k_factor = scaling_factors[k];
if (k_factor == 0)
{
for (std::size_t j = col_begin; j < col_end; ++j)
if (static_cast<int>(Acol_indices[j]) != k )
Avalues[j] = 0.0;

rb[k] = 0.0;
}
else
{
for (std::size_t j = col_begin; j < col_end; ++j)
if(scaling_factors[ Acol_indices[j] ] == 0 )
Avalues[j] = 0.0;
}
}
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

#pragma omp parallel firstprivate(nelements, ElementalDofList)
{
#pragma omp for schedule(guided, 512) nowait
for (int i = 0; i < nelements; i++)
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
#pragma omp for  schedule(guided, 512)
for (int i = 0; i < nconditions; i++)
{
typename ConditionsContainerType::iterator it = rConditions.begin() + i;
const unsigned int this_thread_id = OpenMPUtils::ThisThread();

pScheme->GetConditionDofList(*(it.base()), ElementalDofList, rCurrentProcessInfo);
dofs_aux_list[this_thread_id].insert(ElementalDofList.begin(), ElementalDofList.end());

}
}

if( this->mEchoLevel > 2)
{
KRATOS_INFO("setting_dofs") << "initialize tree reduction" << std::endl;
}

unsigned int old_max = nthreads;
unsigned int new_max = ceil(0.5*static_cast<double>(old_max));
while (new_max>=1 && new_max != old_max)
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
dofs_aux_list[i].insert(dofs_aux_list[i+new_max].begin(), dofs_aux_list[i+new_max].end());
dofs_aux_list[i+new_max].clear();
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
for (auto it= dofs_aux_list[0].begin(); it!= dofs_aux_list[0].end(); it++)
{
Doftemp.push_back( *it );
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
{
omp_destroy_lock(&mlock_array[i]);
}
}
mlock_array.resize(this->mDofSet.size());

for (int i = 0; i < static_cast<int>(mlock_array.size()); i++)
{
omp_init_lock(&mlock_array[i]);
}
#endif

if( this->mEchoLevel > 2)
{
KRATOS_INFO("setting_dofs") << "End of setupdofset" << std::endl;
}

this->Set(LocalFlagType::DOFS_INITIALIZED, true);

KRATOS_CATCH("")
}


void SetUpSystem() override
{

this->mEquationSystemSize = this->mDofSet.size();
int ndofs = static_cast<int>(this->mDofSet.size());

#pragma omp parallel for firstprivate(ndofs)
for (int i = 0; i < static_cast<int>(ndofs); i++)
{
typename DofsArrayType::iterator dof_iterator = this->mDofSet.begin() + i;
dof_iterator->SetEquationId(i);
}

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
KRATOS_WARNING("block builder resize") << "it should not come here -> this is SLOW" << std::endl;
rA.resize(this->mEquationSystemSize, this->mEquationSystemSize, true);
ConstructMatrixStructure(pScheme, rA, rModelPart.Elements(), rModelPart.Conditions(), rModelPart.GetProcessInfo());
}
}
if (rDx.size() != this->mEquationSystemSize)
rDx.resize(this->mEquationSystemSize, false);
if (rb.size() != this->mEquationSystemSize)
rb.resize(this->mEquationSystemSize, false);

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
TSparseSpace::SetToZero(rb);

BuildRHSNoDirichlet(pScheme, rModelPart, rb);

const int ndofs = static_cast<int>(this->mDofSet.size());

#pragma omp parallel for firstprivate(ndofs)
for (int k = 0; k<ndofs; k++)
{
typename DofsArrayType::iterator dof_iterator = this->mDofSet.begin() + k;

const int i = (dof_iterator)->EquationId();
if ( (dof_iterator)->IsFixed() ) {
(dof_iterator)->GetSolutionStepReactionValue() = -rb[i];
} else{
(dof_iterator)->GetSolutionStepReactionValue() = 0.0;
}


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


int Check(ModelPart& r_mUSE_GOOGLE_HASHodel_part) override
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

if (this->mEchoLevel > 1)
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
const std::size_t empty_key = 2*equation_size + 10;
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
for(int iii=0; iii<nelements; iii++)
{
typename ElementsContainerType::iterator i_element = rElements.begin() + iii;
pScheme->EquationId( *(i_element.base()) , ids, rCurrentProcessInfo);
for (std::size_t i = 0; i < ids.size(); i++)
{
#ifdef _OPENMP
omp_set_lock(&mlock_array[ids[i]]);
#endif
auto& row_indices = indices[ids[i]];
row_indices.insert(ids.begin(), ids.end());

#ifdef _OPENMP
omp_unset_lock(&mlock_array[ids[i]]);
#endif
}

}

const int nconditions = static_cast<int>(rConditions.size());
#pragma omp parallel for firstprivate(nconditions, ids)
for (int iii = 0; iii<nconditions; iii++)
{
typename ConditionsContainerType::iterator i_condition = rConditions.begin() + iii;
pScheme->Condition_EquationId( *(i_condition.base()), ids, rCurrentProcessInfo);
for (std::size_t i = 0; i < ids.size(); i++)
{
#ifdef _OPENMP
omp_set_lock(&mlock_array[ids[i]]);
#endif
auto& row_indices = indices[ids[i]];
row_indices.insert(ids.begin(), ids.end());
#ifdef _OPENMP
omp_unset_lock(&mlock_array[ids[i]]);
#endif
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
Arow_indices[i+1] = Arow_indices[i] + indices[i].size();


#pragma omp parallel for
for (int i = 0; i < static_cast<int>(rA.size1()); i++)
{
const unsigned int row_begin = Arow_indices[i];
const unsigned int row_end = Arow_indices[i+1];
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

rA.set_filled(indices.size()+1, nnz);


}


void AssembleLHS(SystemMatrixType& rA,
LocalSystemMatrixType& rLHS_Contribution,
Element::EquationIdVectorType& rEquationId)
{
unsigned int local_size = rLHS_Contribution.size1();

for (unsigned int i_local = 0; i_local < local_size; i_local++)
{
unsigned int i_global = rEquationId[i_local];

for (unsigned int j_local = 0; j_local < local_size; j_local++)
{
unsigned int j_global = rEquationId[j_local];

rA(i_global, j_global) += rLHS_Contribution(i_local, j_local);
}
}

}

void AssembleRHS(SystemVectorType& rb,
LocalSystemVectorType& rRHS_Contribution,
Element::EquationIdVectorType& rEquationId)
{
unsigned int local_size = rRHS_Contribution.size();

for (unsigned int i_local = 0; i_local < local_size; i_local++)
{
unsigned int i_global = rEquationId[i_local];

double& b_value = rb[i_global];
const double& rhs_value = rRHS_Contribution[i_local];

#pragma omp atomic
b_value += rhs_value;
}
}

void Assemble(SystemMatrixType& rA,
SystemVectorType& rb,
const LocalSystemMatrixType& rLHS_Contribution,
const LocalSystemVectorType& rRHS_Contribution,
Element::EquationIdVectorType& rEquationId
#ifdef USE_LOCKS_IN_ASSEMBLY
,std::vector< omp_lock_t >& lock_array
#endif
)
{
unsigned int local_size = rLHS_Contribution.size1();

for (unsigned int i_local = 0; i_local < local_size; i_local++)
{
unsigned int i_global = rEquationId[i_local];

#ifdef USE_LOCKS_IN_ASSEMBLY
omp_set_lock(&lock_array[i_global]);
b[i_global] += rRHS_Contribution(i_local);
#else
double& r_a = rb[i_global];
const double& v_a = rRHS_Contribution(i_local);
#pragma omp atomic
r_a += v_a;
#endif

AssembleRowContribution(rA, rLHS_Contribution, i_global, i_local, rEquationId);

#ifdef USE_LOCKS_IN_ASSEMBLY
omp_unset_lock(&lock_array[i_global]);
#endif
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

void BuildRHSNoDirichlet(SchemePointerType pScheme,
ModelPart& rModelPart,
SystemVectorType& rb)
{
KRATOS_TRY

ElementsContainerType& rElements = rModelPart.Elements();

ConditionsContainerType& ConditionsArray = rModelPart.Conditions();

ProcessInfo& rCurrentProcessInfo = rModelPart.GetProcessInfo();

LocalSystemMatrixType LHS_Contribution = LocalSystemMatrixType(0, 0);
LocalSystemVectorType RHS_Contribution = LocalSystemVectorType(0);

Element::EquationIdVectorType EquationId;


const int nelements = static_cast<int>(rElements.size());
#pragma omp parallel firstprivate(nelements, RHS_Contribution, EquationId)
{
#pragma omp for schedule(guided, 512) nowait
for(int i=0; i<nelements; i++)
{
typename ElementsContainerType::iterator it = rElements.begin() + i;
bool element_is_active = true;
if( (it)->IsDefined(ACTIVE) )
element_is_active = (it)->Is(ACTIVE);

if(element_is_active)
{
pScheme->Calculate_RHS_Contribution(*(it.base()), RHS_Contribution, EquationId, rCurrentProcessInfo);

AssembleRHS(rb, RHS_Contribution, EquationId);
}
}

LHS_Contribution.resize(0, 0, false);
RHS_Contribution.resize(0, false);

const int nconditions = static_cast<int>(ConditionsArray.size());
#pragma omp for schedule(guided, 512)
for (int i = 0; i<nconditions; i++)
{
auto it = ConditionsArray.begin() + i;
bool condition_is_active = true;
if( (it)->IsDefined(ACTIVE) )
condition_is_active = (it)->Is(ACTIVE);

if(condition_is_active)
{
pScheme->Condition_Calculate_RHS_Contribution(*(it.base()), RHS_Contribution, EquationId, rCurrentProcessInfo);

AssembleRHS(rb, RHS_Contribution, EquationId);
}
}
}

KRATOS_CATCH("")
}


inline void CreatePartition(unsigned int number_of_threads,
const int number_of_rows,
vector<unsigned int>& partitions)
{
partitions.resize(number_of_threads + 1);
int partition_size = number_of_rows / number_of_threads;
partitions[0] = 0;
partitions[number_of_threads] = number_of_rows;
for (unsigned int i = 1; i < number_of_threads; i++)
partitions[i] = partitions[i - 1] + partition_size;
}

inline void AssembleRowContribution(SystemMatrixType& rA,
const Matrix& rAlocal,
const unsigned int i,
const unsigned int i_local,
Element::EquationIdVectorType& rEquationId)
{
double* values_vector = rA.value_data().begin();
std::size_t* index1_vector = rA.index1_data().begin();
std::size_t* index2_vector = rA.index2_data().begin();

size_t left_limit = index1_vector[i];

size_t last_pos = ForwardFind(rEquationId[0],left_limit,index2_vector);
size_t last_found = rEquationId[0];

#ifndef USE_LOCKS_IN_ASSEMBLY
double& r_a = values_vector[last_pos];
const double& v_a = rAlocal(i_local,0);
#pragma omp atomic
r_a +=  v_a;
#else
values_vector[last_pos] += rAlocal(i_local,0);
#endif

size_t pos = 0;
for(unsigned int j=1; j<rEquationId.size(); j++)
{
unsigned int id_to_find = rEquationId[j];
if(id_to_find > last_found)
pos = ForwardFind(id_to_find,last_pos+1,index2_vector);
else
pos = BackwardFind(id_to_find,last_pos-1,index2_vector);

#ifndef USE_LOCKS_IN_ASSEMBLY
double& r = values_vector[pos];
const double& v = rAlocal(i_local,j);
#pragma omp atomic
r +=  v;
#else
values_vector[pos] += rAlocal(i_local,j);
#endif

last_found = id_to_find;
last_pos = pos;
}
}

inline unsigned int ForwardFind(const unsigned int id_to_find,
const unsigned int start,
const size_t* index_vector)
{
unsigned int pos = start;
while(id_to_find != index_vector[pos]) pos++;
return pos;
}

inline unsigned int BackwardFind(const unsigned int id_to_find,
const unsigned int start,
const size_t* index_vector)
{
unsigned int pos = start;
while(id_to_find != index_vector[pos]) pos--;
return pos;
}






}; 






}  

#endif 
