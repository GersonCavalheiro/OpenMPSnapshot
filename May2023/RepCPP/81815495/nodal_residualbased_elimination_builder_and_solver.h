
#if !defined(KRATOS_NODAL_RESIDUAL_BASED_ELIMINATION_BUILDER_AND_SOLVER)
#define KRATOS_NODAL_RESIDUAL_BASED_ELIMINATION_BUILDER_AND_SOLVER


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

namespace Kratos
{







template <class TSparseSpace,
class TDenseSpace,  
class TLinearSolver 
>
class NodalResidualBasedEliminationBuilderAndSolver
: public BuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver>
{
public:
KRATOS_CLASS_POINTER_DEFINITION(NodalResidualBasedEliminationBuilderAndSolver);

typedef BuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver> BaseType;

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
typedef GlobalPointersVector<Node> NodeWeakPtrVectorType;



NodalResidualBasedEliminationBuilderAndSolver(
typename TLinearSolver::Pointer pNewLinearSystemSolver)
: BuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver>(pNewLinearSystemSolver)
{
}


~NodalResidualBasedEliminationBuilderAndSolver() override
{
}



void SetMaterialPropertiesToFluid(
ModelPart::NodeIterator itNode,
double &density,
double &deviatoricCoeff,
double &volumetricCoeff,
double timeInterval,
double nodalVolume)
{

deviatoricCoeff = itNode->FastGetSolutionStepValue(DEVIATORIC_COEFFICIENT);
density = itNode->FastGetSolutionStepValue(DENSITY);

volumetricCoeff = timeInterval * itNode->FastGetSolutionStepValue(BULK_MODULUS);

if (volumetricCoeff > 0)
{
volumetricCoeff = timeInterval * itNode->FastGetSolutionStepValue(BULK_MODULUS);
double bulkReduction = density * nodalVolume / (timeInterval * volumetricCoeff);
volumetricCoeff *= bulkReduction;
}
}

void BuildFluidNodally(
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
const ProcessInfo &CurrentProcessInfo = rModelPart.GetProcessInfo();

const unsigned int dimension = rModelPart.ElementsBegin()->GetGeometry().WorkingSpaceDimension();
const double timeInterval = CurrentProcessInfo[DELTA_TIME];
const double FourThirds = 4.0 / 3.0;
const double nTwoThirds = -2.0 / 3.0;

double theta = 0.5;
array_1d<double, 3> Acc(3, 0.0);
double pressure = 0;
double dNdXi = 0;
double dNdYi = 0;
double dNdZi = 0;
double dNdXj = 0;
double dNdYj = 0;
double dNdZj = 0;
unsigned int firstRow = 0;
unsigned int firstCol = 0;

double density = 0;
double deviatoricCoeff = 0;
double volumetricCoeff = 0;


ModelPart::NodeIterator NodesBegin;
ModelPart::NodeIterator NodesEnd;
OpenMPUtils::PartitionedIterators(rModelPart.Nodes(), NodesBegin, NodesEnd);

for (ModelPart::NodeIterator itNode = NodesBegin; itNode != NodesEnd; ++itNode)
{

NodeWeakPtrVectorType &neighb_nodes = itNode->GetValue(NEIGHBOUR_NODES);
Vector nodalSFDneighboursId = itNode->FastGetSolutionStepValue(NODAL_SFD_NEIGHBOURS_ORDER);
const unsigned int neighSize = nodalSFDneighboursId.size();
const double nodalVolume = itNode->FastGetSolutionStepValue(NODAL_VOLUME);

if (neighSize > 1 && nodalVolume > 0)
{

const unsigned int localSize = itNode->FastGetSolutionStepValue(NODAL_SFD_NEIGHBOURS).size();

if (LHS_Contribution.size1() != localSize)
LHS_Contribution.resize(localSize, localSize, false); 

if (RHS_Contribution.size() != localSize)
RHS_Contribution.resize(localSize, false); 

if (EquationId.size() != localSize)
EquationId.resize(localSize, false);

noalias(LHS_Contribution) = ZeroMatrix(localSize, localSize);
noalias(RHS_Contribution) = ZeroVector(localSize);

this->SetMaterialPropertiesToFluid(itNode, density, deviatoricCoeff, volumetricCoeff, timeInterval, nodalVolume);

firstRow = 0;
firstCol = 0;

if (dimension == 2)
{
LHS_Contribution(0, 0) += nodalVolume * density * 2.0 / timeInterval;
LHS_Contribution(1, 1) += nodalVolume * density * 2.0 / timeInterval;

Acc = 2.0 * (itNode->FastGetSolutionStepValue(VELOCITY, 0) - itNode->FastGetSolutionStepValue(VELOCITY, 1)) / timeInterval -
itNode->FastGetSolutionStepValue(ACCELERATION, 0);

RHS_Contribution[0] += -nodalVolume * density * Acc[0];
RHS_Contribution[1] += -nodalVolume * density * Acc[1];


array_1d<double, 3> &VolumeAcceleration = itNode->FastGetSolutionStepValue(VOLUME_ACCELERATION);


RHS_Contribution[0] += nodalVolume * density * VolumeAcceleration[0];
RHS_Contribution[1] += nodalVolume * density * VolumeAcceleration[1];

array_1d<double, 3> Sigma(3, 0.0);
Sigma = itNode->FastGetSolutionStepValue(NODAL_CAUCHY_STRESS);

pressure = itNode->FastGetSolutionStepValue(PRESSURE, 0) * theta + itNode->FastGetSolutionStepValue(PRESSURE, 1) * (1 - theta);
Sigma[0] = itNode->FastGetSolutionStepValue(NODAL_DEVIATORIC_CAUCHY_STRESS)[0] + pressure;
Sigma[1] = itNode->FastGetSolutionStepValue(NODAL_DEVIATORIC_CAUCHY_STRESS)[1] + pressure;

const unsigned int xDofPos = itNode->GetDofPosition(VELOCITY_X);
EquationId[0] = itNode->GetDof(VELOCITY_X, xDofPos).EquationId();
EquationId[1] = itNode->GetDof(VELOCITY_Y, xDofPos + 1).EquationId();

for (unsigned int i = 0; i < neighSize; i++)
{
dNdXi = itNode->FastGetSolutionStepValue(NODAL_SFD_NEIGHBOURS)[firstCol];
dNdYi = itNode->FastGetSolutionStepValue(NODAL_SFD_NEIGHBOURS)[firstCol + 1];

RHS_Contribution[firstCol] += -nodalVolume * (dNdXi * Sigma[0] + dNdYi * Sigma[2]);
RHS_Contribution[firstCol + 1] += -nodalVolume * (dNdYi * Sigma[1] + dNdXi * Sigma[2]);

for (unsigned int j = 0; j < neighSize; j++)
{
dNdXj = itNode->FastGetSolutionStepValue(NODAL_SFD_NEIGHBOURS)[firstRow];
dNdYj = itNode->FastGetSolutionStepValue(NODAL_SFD_NEIGHBOURS)[firstRow + 1];

LHS_Contribution(firstRow, firstCol) += nodalVolume * ((FourThirds * deviatoricCoeff + volumetricCoeff) * dNdXj * dNdXi + dNdYj * dNdYi * deviatoricCoeff) * theta;
LHS_Contribution(firstRow, firstCol + 1) += nodalVolume * ((nTwoThirds * deviatoricCoeff + volumetricCoeff) * dNdXj * dNdYi + dNdYj * dNdXi * deviatoricCoeff) * theta;

LHS_Contribution(firstRow + 1, firstCol) += nodalVolume * ((nTwoThirds * deviatoricCoeff + volumetricCoeff) * dNdYj * dNdXi + dNdXj * dNdYi * deviatoricCoeff) * theta;
LHS_Contribution(firstRow + 1, firstCol + 1) += nodalVolume * ((FourThirds * deviatoricCoeff + volumetricCoeff) * dNdYj * dNdYi + dNdXj * dNdXi * deviatoricCoeff) * theta;

firstRow += 2;
}

firstRow = 0;
firstCol += 2;

if (i < neighb_nodes.size())
{
EquationId[firstCol] = neighb_nodes[i].GetDof(VELOCITY_X, xDofPos).EquationId();
EquationId[firstCol + 1] = neighb_nodes[i].GetDof(VELOCITY_Y, xDofPos + 1).EquationId();
}
}
}
else if (dimension == 3)
{
LHS_Contribution(0, 0) += nodalVolume * density * 2.0 / timeInterval;
LHS_Contribution(1, 1) += nodalVolume * density * 2.0 / timeInterval;
LHS_Contribution(2, 2) += nodalVolume * density * 2.0 / timeInterval;

Acc = 2.0 * (itNode->FastGetSolutionStepValue(VELOCITY, 0) - itNode->FastGetSolutionStepValue(VELOCITY, 1)) / timeInterval -
itNode->FastGetSolutionStepValue(ACCELERATION, 0);

RHS_Contribution[0] += -nodalVolume * density * Acc[0];
RHS_Contribution[1] += -nodalVolume * density * Acc[1];
RHS_Contribution[2] += -nodalVolume * density * Acc[2];


array_1d<double, 3> &VolumeAcceleration = itNode->FastGetSolutionStepValue(VOLUME_ACCELERATION);

RHS_Contribution[0] += nodalVolume * density * VolumeAcceleration[0];
RHS_Contribution[1] += nodalVolume * density * VolumeAcceleration[1];
RHS_Contribution[2] += nodalVolume * density * VolumeAcceleration[2];


array_1d<double, 6> Sigma(6, 0.0);
Sigma = itNode->FastGetSolutionStepValue(NODAL_CAUCHY_STRESS);

pressure = itNode->FastGetSolutionStepValue(PRESSURE, 0) * theta + itNode->FastGetSolutionStepValue(PRESSURE, 1) * (1 - theta);
Sigma[0] = itNode->FastGetSolutionStepValue(NODAL_DEVIATORIC_CAUCHY_STRESS)[0] + pressure;
Sigma[1] = itNode->FastGetSolutionStepValue(NODAL_DEVIATORIC_CAUCHY_STRESS)[1] + pressure;
Sigma[2] = itNode->FastGetSolutionStepValue(NODAL_DEVIATORIC_CAUCHY_STRESS)[2] + pressure;

const unsigned int xDofPos = itNode->GetDofPosition(VELOCITY_X);
EquationId[0] = itNode->GetDof(VELOCITY_X, xDofPos).EquationId();
EquationId[1] = itNode->GetDof(VELOCITY_Y, xDofPos + 1).EquationId();
EquationId[2] = itNode->GetDof(VELOCITY_Z, xDofPos + 2).EquationId();

for (unsigned int i = 0; i < neighSize; i++)
{
dNdXi = itNode->FastGetSolutionStepValue(NODAL_SFD_NEIGHBOURS)[firstCol];
dNdYi = itNode->FastGetSolutionStepValue(NODAL_SFD_NEIGHBOURS)[firstCol + 1];
dNdZi = itNode->FastGetSolutionStepValue(NODAL_SFD_NEIGHBOURS)[firstCol + 2];

RHS_Contribution[firstCol] += -nodalVolume * (dNdXi * Sigma[0] + dNdYi * Sigma[3] + dNdZi * Sigma[4]);
RHS_Contribution[firstCol + 1] += -nodalVolume * (dNdYi * Sigma[1] + dNdXi * Sigma[3] + dNdZi * Sigma[5]);
RHS_Contribution[firstCol + 2] += -nodalVolume * (dNdZi * Sigma[2] + dNdXi * Sigma[4] + dNdYi * Sigma[5]);

for (unsigned int j = 0; j < neighSize; j++)
{

dNdXj = itNode->FastGetSolutionStepValue(NODAL_SFD_NEIGHBOURS)[firstRow];
dNdYj = itNode->FastGetSolutionStepValue(NODAL_SFD_NEIGHBOURS)[firstRow + 1];
dNdZj = itNode->FastGetSolutionStepValue(NODAL_SFD_NEIGHBOURS)[firstRow + 2];

LHS_Contribution(firstRow, firstCol) += nodalVolume * ((FourThirds * deviatoricCoeff + volumetricCoeff) * dNdXj * dNdXi + (dNdYj * dNdYi + dNdZj * dNdZi) * deviatoricCoeff) * theta;
LHS_Contribution(firstRow, firstCol + 1) += nodalVolume * ((nTwoThirds * deviatoricCoeff + volumetricCoeff) * dNdXj * dNdYi + dNdYj * dNdXi * deviatoricCoeff) * theta;
LHS_Contribution(firstRow, firstCol + 2) += nodalVolume * ((nTwoThirds * deviatoricCoeff + volumetricCoeff) * dNdXj * dNdZi + dNdZj * dNdXi * deviatoricCoeff) * theta;

LHS_Contribution(firstRow + 1, firstCol) += nodalVolume * ((nTwoThirds * deviatoricCoeff + volumetricCoeff) * dNdYj * dNdXi + dNdXj * dNdYi * deviatoricCoeff) * theta;
LHS_Contribution(firstRow + 1, firstCol + 1) += nodalVolume * ((FourThirds * deviatoricCoeff + volumetricCoeff) * dNdYj * dNdYi + (dNdXj * dNdXi + dNdZj * dNdZi) * deviatoricCoeff) * theta;
LHS_Contribution(firstRow + 1, firstCol + 2) += nodalVolume * ((nTwoThirds * deviatoricCoeff + volumetricCoeff) * dNdYj * dNdZi + dNdZj * dNdYi * deviatoricCoeff) * theta;

LHS_Contribution(firstRow + 2, firstCol) += nodalVolume * ((nTwoThirds * deviatoricCoeff + volumetricCoeff) * dNdZj * dNdXi + dNdXj * dNdZi * deviatoricCoeff) * theta;
LHS_Contribution(firstRow + 2, firstCol + 1) += nodalVolume * ((nTwoThirds * deviatoricCoeff + volumetricCoeff) * dNdZj * dNdYi + dNdYj * dNdZi * deviatoricCoeff) * theta;
LHS_Contribution(firstRow + 2, firstCol + 2) += nodalVolume * ((FourThirds * deviatoricCoeff + volumetricCoeff) * dNdZj * dNdZi + (dNdXj * dNdXi + dNdYj * dNdYi) * deviatoricCoeff) * theta;

firstRow += 3;
}

firstRow = 0;
firstCol += 3;

if (i < neighb_nodes.size())
{
EquationId[firstCol] = neighb_nodes[i].GetDof(VELOCITY_X, xDofPos).EquationId();
EquationId[firstCol + 1] = neighb_nodes[i].GetDof(VELOCITY_Y, xDofPos + 1).EquationId();
EquationId[firstCol + 2] = neighb_nodes[i].GetDof(VELOCITY_Z, xDofPos + 2).EquationId();
}
}
}

#ifdef _OPENMP
Assemble(A, b, LHS_Contribution, RHS_Contribution, EquationId, mlock_array);
#else
Assemble(A, b, LHS_Contribution, RHS_Contribution, EquationId);
#endif
}
}


KRATOS_CATCH("")
}


void SystemSolve(
TSystemMatrixType &A,
TSystemVectorType &Dx,
TSystemVectorType &b) override
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

KRATOS_INFO_IF("NodalResidualBasedEliminationBuilderAndSolver", this->GetEchoLevel() > 1) << *(BaseType::mpLinearSystemSolver) << std::endl;

KRATOS_CATCH("")
}


void SystemSolveWithPhysics(
TSystemMatrixType &A,
TSystemVectorType &Dx,
TSystemVectorType &b,
ModelPart &rModelPart)
{
KRATOS_TRY

double norm_b;
if (TSparseSpace::Size(b) != 0)
norm_b = TSparseSpace::TwoNorm(b);
else
norm_b = 0.00;

if (norm_b != 0.00)
{
if (BaseType::mpLinearSystemSolver->AdditionalPhysicalDataIsNeeded())
BaseType::mpLinearSystemSolver->ProvideAdditionalData(A, Dx, b, BaseType::mDofSet, rModelPart);

BaseType::mpLinearSystemSolver->Solve(A, Dx, b);
}
else
{
TSparseSpace::SetToZero(Dx);
KRATOS_WARNING_IF("NodalResidualBasedEliminationBuilderAndSolver", rModelPart.GetCommunicator().MyPID() == 0) << "ATTENTION! setting the RHS to zero!" << std::endl;
}

KRATOS_INFO_IF("NodalResidualBasedEliminationBuilderAndSolver", this->GetEchoLevel() > 1 && rModelPart.GetCommunicator().MyPID() == 0) << *(BaseType::mpLinearSystemSolver) << std::endl;

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
BuildFluidNodally(pScheme, rModelPart, A, b);
Timer::Stop("Build");


ApplyDirichletConditions(pScheme, rModelPart, A, Dx, b);

KRATOS_INFO_IF("ResidualBasedBlockBuilderAndSolver", (this->GetEchoLevel() == 3)) << "Before the solution of the system"
<< "\nSystem Matrix = " << A << "\nUnknowns vector = " << Dx << "\nRHS vector = " << b << std::endl;

SystemSolveWithPhysics(A, Dx, b, rModelPart);

KRATOS_INFO_IF("ResidualBasedBlockBuilderAndSolver", (this->GetEchoLevel() == 3)) << "After the solution of the system"
<< "\nSystem Matrix = " << A << "\nUnknowns vector = " << Dx << "\nRHS vector = " << b << std::endl;

KRATOS_CATCH("")
}


void SetUpDofSet(
typename TSchemeType::Pointer pScheme,
ModelPart &rModelPart) override
{
KRATOS_TRY;

KRATOS_INFO_IF("NodalResidualBasedEliminationBuilderAndSolver", this->GetEchoLevel() > 1 && rModelPart.GetCommunicator().MyPID() == 0) << "Setting up the dofs" << std::endl;

ElementsArrayType &pElements = rModelPart.Elements();
const int nelements = static_cast<int>(pElements.size());

Element::DofsVectorType ElementalDofList;

const ProcessInfo &CurrentProcessInfo = rModelPart.GetProcessInfo();

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

ConditionsArrayType &pConditions = rModelPart.Conditions();
const int nconditions = static_cast<int>(pConditions.size());
#pragma omp parallel for firstprivate(nconditions, ElementalDofList)
for (int i = 0; i < nconditions; ++i)
{
auto it_cond = pConditions.begin() + i;
const IndexType this_thread_id = OpenMPUtils::ThisThread();

pScheme->GetDofList(*it_cond, ElementalDofList, CurrentProcessInfo);
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
Doftemp.push_back((*it));
}
Doftemp.Sort();

BaseType::mDofSet = Doftemp;

KRATOS_ERROR_IF(BaseType::mDofSet.size() == 0) << "No degrees of freedom!" << std::endl;

BaseType::mDofSetIsInitialized = true;

KRATOS_INFO_IF("NodalResidualBasedEliminationBuilderAndSolver", this->GetEchoLevel() > 2 && rModelPart.GetCommunicator().MyPID() == 0) << "Finished setting up the dofs" << std::endl;

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


void SetUpSystem(
ModelPart &rModelPart) override
{
int free_id = 0;
int fix_id = BaseType::mDofSet.size();

for (typename DofsArrayType::iterator dof_iterator = BaseType::mDofSet.begin(); dof_iterator != BaseType::mDofSet.end(); ++dof_iterator)
if (dof_iterator->IsFixed())
dof_iterator->SetEquationId(--fix_id);
else
dof_iterator->SetEquationId(free_id++);

BaseType::mEquationSystemSize = fix_id;
}

/
void ApplyDirichletConditions(
typename TSchemeType::Pointer pScheme,
ModelPart &rModelPart,
TSystemMatrixType &A,
TSystemVectorType &Dx,
TSystemVectorType &b) override
{
}


void Clear() override
{
this->mDofSet = DofsArrayType();

if (this->mpReactionsVector != NULL)
TSparseSpace::Clear((this->mpReactionsVector));

this->mpLinearSystemSolver->Clear();

KRATOS_INFO_IF("NodalResidualBasedEliminationBuilderAndSolver", this->GetEchoLevel() > 1) << "Clear Function called" << std::endl;
}


int Check(ModelPart &rModelPart) override
{
KRATOS_TRY

return 0;
KRATOS_CATCH("");
}





protected:




void Assemble(
TSystemMatrixType &A,
TSystemVectorType &b,
const LocalSystemMatrixType &LHS_Contribution,
const LocalSystemVectorType &RHS_Contribution,
const Element::EquationIdVectorType &EquationId
#ifdef _OPENMP
,
std::vector<omp_lock_t> &lock_array
#endif
)
{
unsigned int local_size = LHS_Contribution.size1();

for (unsigned int i_local = 0; i_local < local_size; i_local++)
{
unsigned int i_global = EquationId[i_local];

if (i_global < BaseType::mEquationSystemSize)
{
#ifdef _OPENMP
omp_set_lock(&lock_array[i_global]);
#endif
b[i_global] += RHS_Contribution(i_local);
for (unsigned int j_local = 0; j_local < local_size; j_local++)
{
unsigned int j_global = EquationId[j_local];
if (j_global < BaseType::mEquationSystemSize)
{
A(i_global, j_global) += LHS_Contribution(i_local, j_local);
}
}
#ifdef _OPENMP
omp_unset_lock(&lock_array[i_global]);
#endif
}
}
}

/




} 

#endif 
