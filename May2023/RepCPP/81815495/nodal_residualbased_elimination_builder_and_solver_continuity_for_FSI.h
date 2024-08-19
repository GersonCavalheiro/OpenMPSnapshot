
#if !defined(KRATOS_NODAL_RESIDUAL_BASED_ELIMINATION_BUILDER_AND_SOLVER_CONTINUITY_FOR_FSI)
#define KRATOS_NODAL_RESIDUAL_BASED_ELIMINATION_BUILDER_AND_SOLVER_CONTINUITY_FOR_FSI


#include <set>
#ifdef _OPENMP
#include <omp.h>
#endif


#ifdef USE_GOOGLE_HASH
#include "sparsehash/dense_hash_set" 
#else
#include <unordered_set>
#endif


#include "utilities/timer.h"
#include "includes/define.h"
#include "includes/key_hash.h"
#include "solving_strategies/builder_and_solvers/builder_and_solver.h"
#include "includes/model_part.h"

#include "pfem_fluid_dynamics_application_variables.h"
#include "nodal_residualbased_elimination_builder_and_solver_continuity.h"

namespace Kratos
{







template <class TSparseSpace,
class TDenseSpace,  
class TLinearSolver 
>
class NodalResidualBasedEliminationBuilderAndSolverContinuityForFSI
: public NodalResidualBasedEliminationBuilderAndSolverContinuity<TSparseSpace, TDenseSpace, TLinearSolver>
{
public:
KRATOS_CLASS_POINTER_DEFINITION(NodalResidualBasedEliminationBuilderAndSolverContinuityForFSI);

typedef NodalResidualBasedEliminationBuilderAndSolverContinuity<TSparseSpace, TDenseSpace, TLinearSolver> BaseType;

typedef typename BaseType::TSchemeType TSchemeType;

typedef typename BaseType::TDataType TDataType;

typedef typename BaseType::DofsArrayType DofsArrayType;

typedef typename BaseType::TSystemMatrixType TSystemMatrixType;

typedef typename BaseType::TSystemVectorType TSystemVectorType;

typedef typename BaseType::LocalSystemVectorType LocalSystemVectorType;

typedef typename BaseType::LocalSystemMatrixType LocalSystemMatrixType;

typedef typename BaseType::TSystemMatrixPointerType TSystemMatrixPointerType;
typedef typename BaseType::TSystemVectorPointerType TSystemVectorPointerType;

typedef Node NodeType;

typedef typename BaseType::NodesArrayType NodesArrayType;
typedef typename BaseType::ElementsArrayType ElementsArrayType;
typedef typename BaseType::ConditionsArrayType ConditionsArrayType;

typedef typename BaseType::ElementsContainerType ElementsContainerType;

typedef Vector VectorType;



NodalResidualBasedEliminationBuilderAndSolverContinuityForFSI(
typename TLinearSolver::Pointer pNewLinearSystemSolver)
: NodalResidualBasedEliminationBuilderAndSolverContinuity<TSparseSpace, TDenseSpace, TLinearSolver>(pNewLinearSystemSolver)
{
}


~NodalResidualBasedEliminationBuilderAndSolverContinuityForFSI() override
{
}



void BuildAll(
typename TSchemeType::Pointer pScheme,
ModelPart &rModelPart,
TSystemMatrixType &A,
TSystemVectorType &b)
{
KRATOS_TRY

KRATOS_ERROR_IF(!pScheme) << "No scheme provided!" << std::endl;

LocalSystemMatrixType LHS_Contribution = LocalSystemMatrixType(0, 0);
LocalSystemVectorType RHS_Contribution = LocalSystemVectorType(0);

Element::EquationIdVectorType EquationId;

ProcessInfo &CurrentProcessInfo = rModelPart.GetProcessInfo();

const double timeInterval = CurrentProcessInfo[DELTA_TIME];
const double deviatoric_threshold = 0.1;
double deltaPressure = 0;


ModelPart::NodeIterator NodesBegin;
ModelPart::NodeIterator NodesEnd;
OpenMPUtils::PartitionedIterators(rModelPart.Nodes(), NodesBegin, NodesEnd);

for (ModelPart::NodeIterator itNode = NodesBegin; itNode != NodesEnd; ++itNode)
{

NodeWeakPtrVectorType &neighb_nodes = itNode->GetValue(NEIGHBOUR_NODES);
const unsigned int neighSize = neighb_nodes.size() + 1;

if (neighSize > 1)
{

if (LHS_Contribution.size1() != 1)
LHS_Contribution.resize(1, 1, false); 

if (RHS_Contribution.size() != 1)
RHS_Contribution.resize(1, false); 

noalias(LHS_Contribution) = ZeroMatrix(1, 1);
noalias(RHS_Contribution) = ZeroVector(1);

if (EquationId.size() != 1)
EquationId.resize(1, false);

if ((itNode->Is(FLUID) && itNode->IsNot(SOLID)) || itNode->FastGetSolutionStepValue(INTERFACE_NODE) == true)
{

double nodalVolume = itNode->FastGetSolutionStepValue(NODAL_VOLUME);

if (nodalVolume > 0)
{ 

double deviatoricCoeff = 0;
this->GetDeviatoricCoefficientForFluid(rModelPart, itNode, deviatoricCoeff);

if (deviatoricCoeff > deviatoric_threshold && itNode->IsNot(SOLID))
{
deviatoricCoeff = deviatoric_threshold;
}

double volumetricCoeff = itNode->FastGetSolutionStepValue(VOLUMETRIC_COEFFICIENT) + 2.0 * deviatoricCoeff / 3.0;
if (itNode->IsNot(SOLID) || itNode->FastGetSolutionStepValue(INTERFACE_NODE) == true)
{
volumetricCoeff = timeInterval * itNode->FastGetSolutionStepValue(BULK_MODULUS);
}

deltaPressure = itNode->FastGetSolutionStepValue(PRESSURE, 0) - itNode->FastGetSolutionStepValue(PRESSURE, 1);

LHS_Contribution(0, 0) += nodalVolume / volumetricCoeff;

RHS_Contribution[0] += -deltaPressure * nodalVolume / volumetricCoeff;

RHS_Contribution[0] += itNode->GetSolutionStepValue(NODAL_VOLUMETRIC_DEF_RATE) * nodalVolume;
}
}

if (itNode->Is(SOLID))
{
double nodalVolume = itNode->FastGetSolutionStepValue(SOLID_NODAL_VOLUME);
double youngModulus = itNode->FastGetSolutionStepValue(YOUNG_MODULUS);
double poissonRatio = itNode->FastGetSolutionStepValue(POISSON_RATIO);

double deviatoricCoeff = timeInterval * youngModulus / (1.0 + poissonRatio) * 0.5;
double volumetricCoeff = timeInterval * poissonRatio * youngModulus / ((1.0 + poissonRatio) * (1.0 - 2.0 * poissonRatio)) + 2.0 * deviatoricCoeff / 3.0;

deltaPressure = itNode->FastGetSolutionStepValue(PRESSURE, 0) - itNode->FastGetSolutionStepValue(PRESSURE, 1);

LHS_Contribution(0, 0) += nodalVolume / volumetricCoeff;

RHS_Contribution[0] += -deltaPressure * nodalVolume / volumetricCoeff;

RHS_Contribution[0] += itNode->GetSolutionStepValue(SOLID_NODAL_VOLUMETRIC_DEF_RATE) * nodalVolume;
}

const unsigned int xDofPos = itNode->GetDofPosition(PRESSURE);

EquationId[0] = itNode->GetDof(PRESSURE, xDofPos).EquationId();

#ifdef _OPENMP
Assemble(A, b, LHS_Contribution, RHS_Contribution, EquationId, mlock_array);
#else
Assemble(A, b, LHS_Contribution, RHS_Contribution, EquationId);
#endif
}
}


ElementsArrayType &pElements = rModelPart.Elements();
int number_of_threads = ParallelUtilities::GetNumThreads();

#ifdef _OPENMP
int A_size = A.size1();

std::vector<omp_lock_t> lock_array(A.size1());

for (int i = 0; i < A_size; i++)
omp_init_lock(&lock_array[i]);
#endif

DenseVector<unsigned int> element_partition;
CreatePartition(number_of_threads, pElements.size(), element_partition);
if (this->GetEchoLevel() > 0)
{
KRATOS_WATCH(number_of_threads);
KRATOS_WATCH(element_partition);
}

#pragma omp parallel for firstprivate(number_of_threads) schedule(static, 1)
for (int k = 0; k < number_of_threads; k++)
{
LocalSystemMatrixType elementalLHS_Contribution = LocalSystemMatrixType(0, 0);
LocalSystemVectorType elementalRHS_Contribution = LocalSystemVectorType(0);

Element::EquationIdVectorType elementalEquationId;
const ProcessInfo &CurrentProcessInfo = rModelPart.GetProcessInfo();
typename ElementsArrayType::ptr_iterator it_begin = pElements.ptr_begin() + element_partition[k];
typename ElementsArrayType::ptr_iterator it_end = pElements.ptr_begin() + element_partition[k + 1];

unsigned int pos = (rModelPart.Nodes().begin())->GetDofPosition(PRESSURE);

for (typename ElementsArrayType::ptr_iterator it = it_begin; it != it_end; ++it)
{
if ((*it)->IsNot(SOLID))
{
(*it)->CalculateLocalSystem(elementalLHS_Contribution, elementalRHS_Contribution, CurrentProcessInfo);

Geometry<Node> &geom = (*it)->GetGeometry();
if (elementalEquationId.size() != geom.size())
elementalEquationId.resize(geom.size(), false);

for (unsigned int i = 0; i < geom.size(); i++)
elementalEquationId[i] = geom[i].GetDof(PRESSURE, pos).EquationId();

#ifdef _OPENMP
this->Assemble(A, b, elementalLHS_Contribution, elementalRHS_Contribution, elementalEquationId, lock_array);
#else
this->Assemble(A, b, elementalLHS_Contribution, elementalRHS_Contribution, elementalEquationId);
#endif
}
}
}

#ifdef _OPENMP
for (int i = 0; i < A_size; i++)
omp_destroy_lock(&lock_array[i]);
#endif

KRATOS_CATCH("")
}


void BuildAndSolve(
typename TSchemeType::Pointer pScheme,
ModelPart &rModelPart,
TSystemMatrixType &A,
TSystemVectorType &Dx,
TSystemVectorType &b) override
{
KRATOS_TRY

Timer::Start("Build");



BuildAll(pScheme, rModelPart, A, b);

Timer::Stop("Build");


this->ApplyDirichletConditions(pScheme, rModelPart, A, Dx, b);

KRATOS_INFO_IF("ResidualBasedBlockBuilderAndSolver", (this->GetEchoLevel() == 3)) << "Before the solution of the system"
<< "\nSystem Matrix = " << A << "\nUnknowns vector = " << Dx << "\nRHS vector = " << b << std::endl;

this->SystemSolveWithPhysics(A, Dx, b, rModelPart);

KRATOS_INFO_IF("ResidualBasedBlockBuilderAndSolver", (this->GetEchoLevel() == 3)) << "After the solution of the system"
<< "\nSystem Matrix = " << A << "\nUnknowns vector = " << Dx << "\nRHS vector = " << b << std::endl;

KRATOS_CATCH("")
}


void SetUpDofSet(
typename TSchemeType::Pointer pScheme,
ModelPart &rModelPart) override
{
KRATOS_TRY;

KRATOS_INFO_IF("NodalResidualBasedEliminationBuilderAndSolverContinuityForFSI", this->GetEchoLevel() > 1 && rModelPart.GetCommunicator().MyPID() == 0) << "Setting up the dofs" << std::endl;

ElementsArrayType &pElements = rModelPart.Elements();
const int nelements = static_cast<int>(pElements.size());

Element::DofsVectorType ElementalDofList;

ProcessInfo &CurrentProcessInfo = rModelPart.GetProcessInfo();

unsigned int nthreads = ParallelUtilities::GetNumThreads();


#ifdef USE_GOOGLE_HASH
typedef google::dense_hash_set<NodeType::DofType::Pointer, DofPointerHasher> set_type;
#else
typedef std::unordered_set<NodeType::DofType::Pointer, DofPointerHasher> set_type;
#endif

std::vector<set_type> dofs_aux_list(nthreads);

for (int i = 0; i < static_cast<int>(nthreads); i++)
{
#ifdef USE_GOOGLE_HASH
dofs_aux_list[i].set_empty_key(NodeType::DofType::Pointer());
#else
dofs_aux_list[i].reserve(nelements);
#endif
}

for (int i = 0; i < static_cast<int>(nelements); ++i)
{
auto it_elem = pElements.begin() + i;
const IndexType this_thread_id = OpenMPUtils::ThisThread();

pScheme->GetDofList(*it_elem, ElementalDofList, CurrentProcessInfo);

dofs_aux_list[this_thread_id].insert(ElementalDofList.begin(), ElementalDofList.end());
}

unsigned int old_max = nthreads;
unsigned int new_max = ceil(0.5 * static_cast<double>(old_max));
while (new_max >= 1 && new_max != old_max)
{

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
new_max = ceil(0.5 * static_cast<double>(old_max));
}

DofsArrayType Doftemp;
BaseType::mDofSet = DofsArrayType();

Doftemp.reserve(dofs_aux_list[0].size());
for (auto it = dofs_aux_list[0].begin(); it != dofs_aux_list[0].end(); it++)
{
Doftemp.push_back(*it);
}
Doftemp.Sort();

BaseType::mDofSet = Doftemp;

KRATOS_ERROR_IF(BaseType::mDofSet.size() == 0) << "No degrees of freedom!" << std::endl;

BaseType::mDofSetIsInitialized = true;

KRATOS_INFO_IF("NodalResidualBasedEliminationBuilderAndSolverContinuityForFSI", this->GetEchoLevel() > 2 && rModelPart.GetCommunicator().MyPID() == 0) << "Finished setting up the dofs" << std::endl;

#ifdef _OPENMP
if (mlock_array.size() != 0)
{
for (int i = 0; i < static_cast<int>(mlock_array.size()); i++)
omp_destroy_lock(&mlock_array[i]);
}

mlock_array.resize(BaseType::mDofSet.size());

for (int i = 0; i < static_cast<int>(mlock_array.size()); i++)
omp_init_lock(&mlock_array[i]);
#endif

#ifdef KRATOS_DEBUG
if (BaseType::GetCalculateReactionsFlag())
{
for (auto dof_iterator = BaseType::mDofSet.begin(); dof_iterator != BaseType::mDofSet.end(); ++dof_iterator)
{
KRATOS_ERROR_IF_NOT(dof_iterator->HasReaction()) << "Reaction variable not set for the following : " << std::endl
<< "Node : " << dof_iterator->Id() << std::endl
<< "Dof : " << (*dof_iterator) << std::endl
<< "Not possible to calculate reactions." << std::endl;
}
}
#endif

KRATOS_CATCH("");
}

/




} 

#endif 
