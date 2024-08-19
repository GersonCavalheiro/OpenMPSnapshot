
#pragma once



#include "solving_strategies/strategies/implicit_solving_strategy.h"
#include "utilities/builtin_timer.h"
#include "utilities/atomic_utilities.h"
#include "utilities/entities_utilities.h"

#include "structural_mechanics_application_variables.h"

namespace Kratos
{






template<class TSparseSpace,
class TDenseSpace,
class TLinearSolver
>
class EigensolverStrategy
: public ImplicitSolvingStrategy<TSparseSpace, TDenseSpace, TLinearSolver>
{
public:

KRATOS_CLASS_POINTER_DEFINITION(EigensolverStrategy);

typedef ImplicitSolvingStrategy<TSparseSpace, TDenseSpace, TLinearSolver> BaseType;

typedef typename BaseType::TSchemeType::Pointer SchemePointerType;

typedef typename BaseType::TBuilderAndSolverType::Pointer BuilderAndSolverPointerType;

typedef typename TDenseSpace::VectorType DenseVectorType;

typedef typename TDenseSpace::MatrixType DenseMatrixType;

typedef TSparseSpace SparseSpaceType;

typedef typename TSparseSpace::VectorPointerType SparseVectorPointerType;

typedef typename TSparseSpace::MatrixPointerType SparseMatrixPointerType;

typedef typename TSparseSpace::MatrixType SparseMatrixType;

typedef typename TSparseSpace::VectorType SparseVectorType;


EigensolverStrategy(
ModelPart& rModelPart,
SchemePointerType pScheme,
BuilderAndSolverPointerType pBuilderAndSolver,
double MassMatrixDiagonalValue,
double StiffnessMatrixDiagonalValue,
bool ComputeModalDecomposition = false
)
: ImplicitSolvingStrategy<TSparseSpace, TDenseSpace, TLinearSolver>(rModelPart),
mMassMatrixDiagonalValue(MassMatrixDiagonalValue),
mStiffnessMatrixDiagonalValue(StiffnessMatrixDiagonalValue),
mComputeModalDecompostion(ComputeModalDecomposition)
{
KRATOS_TRY

mpScheme = pScheme;

mpBuilderAndSolver = pBuilderAndSolver;

mpBuilderAndSolver->SetDofSetIsInitializedFlag(false);

this->SetEchoLevel(0);

this->SetRebuildLevel(1);

SparseMatrixType* AuxMassMatrix = new SparseMatrixType;
mpMassMatrix = Kratos::shared_ptr<SparseMatrixType>(AuxMassMatrix);
SparseMatrixType* AuxStiffnessMatrix = new SparseMatrixType;
mpStiffnessMatrix = Kratos::shared_ptr<SparseMatrixType>(AuxStiffnessMatrix);

KRATOS_CATCH("")
}

EigensolverStrategy(const EigensolverStrategy& Other) = delete;

~EigensolverStrategy() override
{
this->Clear();
}



void SetIsInitialized(bool val)
{
mInitializeWasPerformed = val;
}

bool GetIsInitialized() const
{
return mInitializeWasPerformed;
}

void SetScheme(SchemePointerType pScheme)
{
mpScheme = pScheme;
};

SchemePointerType& pGetScheme()
{
return mpScheme;
};

void SetBuilderAndSolver(BuilderAndSolverPointerType pNewBuilderAndSolver)
{
mpBuilderAndSolver = pNewBuilderAndSolver;
};

BuilderAndSolverPointerType& pGetBuilderAndSolver()
{
return mpBuilderAndSolver;
};

SparseMatrixType& GetMassMatrix()
{
return *mpMassMatrix;
}

SparseMatrixType& GetStiffnessMatrix()
{
return *mpStiffnessMatrix;
}

SparseMatrixPointerType& pGetMassMatrix()
{
return mpMassMatrix;
}

SparseMatrixPointerType& pGetStiffnessMatrix()
{
return mpStiffnessMatrix;
}

void SetReformDofSetAtEachStepFlag(bool flag)
{
this->pGetBuilderAndSolver()->SetReshapeMatrixFlag(flag);
}

bool GetReformDofSetAtEachStepFlag() const
{
return this->pGetBuilderAndSolver()->GetReshapeMatrixFlag();
}


void SetEchoLevel(int Level) override
{
BaseType::SetEchoLevel(Level);
this->pGetBuilderAndSolver()->SetEchoLevel(Level);
}


void Initialize() override
{
KRATOS_TRY

ModelPart& rModelPart = BaseType::GetModelPart();
const int rank = rModelPart.GetCommunicator().MyPID();

KRATOS_INFO_IF("EigensolverStrategy", BaseType::GetEchoLevel() > 2 && rank == 0)
<<  "Entering Initialize" << std::endl;

if (mInitializeWasPerformed == false)
{
SchemePointerType& pScheme = this->pGetScheme();

if (pScheme->SchemeIsInitialized() == false)
pScheme->Initialize(rModelPart);

if (pScheme->ElementsAreInitialized() == false)
pScheme->InitializeElements(rModelPart);

if (pScheme->ConditionsAreInitialized() == false)
pScheme->InitializeConditions(rModelPart);
}

KRATOS_INFO_IF("EigensolverStrategy", BaseType::GetEchoLevel() > 2 && rank == 0)
<<  "Exiting Initialize" << std::endl;

KRATOS_CATCH("")
}


void Clear() override
{
KRATOS_TRY

BuilderAndSolverPointerType& pBuilderAndSolver = this->pGetBuilderAndSolver();
pBuilderAndSolver->GetLinearSystemSolver()->Clear();

if (this->pGetMassMatrix() != nullptr)
this->pGetMassMatrix() = nullptr;

if (this->pGetStiffnessMatrix() != nullptr)
this->pGetStiffnessMatrix() = nullptr;

pBuilderAndSolver->SetDofSetIsInitializedFlag(false);

pBuilderAndSolver->Clear();

this->pGetScheme()->Clear();

mInitializeWasPerformed = false;

KRATOS_CATCH("")
}


void InitializeSolutionStep() override
{
KRATOS_TRY

ModelPart& rModelPart = BaseType::GetModelPart();
const int rank = rModelPart.GetCommunicator().MyPID();

KRATOS_INFO_IF("EigensolverStrategy", BaseType::GetEchoLevel() > 2 && rank == 0)
<<  "Entering InitializeSolutionStep" << std::endl;

BuilderAndSolverPointerType& pBuilderAndSolver = this->pGetBuilderAndSolver();
SchemePointerType& pScheme = this->pGetScheme();
SparseMatrixPointerType& pStiffnessMatrix = this->pGetStiffnessMatrix();
SparseMatrixType& rStiffnessMatrix = this->GetStiffnessMatrix();

SparseVectorPointerType pDx = SparseSpaceType::CreateEmptyVectorPointer();
SparseVectorPointerType pb = SparseSpaceType::CreateEmptyVectorPointer();
auto& rDx = *pDx;
auto& rb = *pb;

BuiltinTimer system_construction_time;
if (pBuilderAndSolver->GetDofSetIsInitializedFlag() == false ||
pBuilderAndSolver->GetReshapeMatrixFlag() == true)
{
BuiltinTimer setup_dofs_time;
pBuilderAndSolver->SetUpDofSet(pScheme, rModelPart);

KRATOS_INFO_IF("Setup Dofs Time", BaseType::GetEchoLevel() > 0 && rank == 0)
<< setup_dofs_time.ElapsedSeconds() << std::endl;

BuiltinTimer setup_system_time;
pBuilderAndSolver->SetUpSystem(rModelPart);

KRATOS_INFO_IF("Setup System Time", BaseType::GetEchoLevel() > 0 && rank == 0)
<< setup_system_time.ElapsedSeconds() << std::endl;

BuiltinTimer system_matrix_resize_time;
SparseMatrixPointerType& pMassMatrix = this->pGetMassMatrix();

pBuilderAndSolver->ResizeAndInitializeVectors(
pScheme, pMassMatrix, pDx, pb, rModelPart);

pBuilderAndSolver->ResizeAndInitializeVectors(
pScheme, pStiffnessMatrix, pDx, pb, rModelPart);

KRATOS_INFO_IF("System Matrix Resize Time", BaseType::GetEchoLevel() > 0 && rank == 0)
<< system_matrix_resize_time.ElapsedSeconds() << std::endl;
}
else
{
SparseSpaceType::Resize(rb, SparseSpaceType::Size1(rStiffnessMatrix));
SparseSpaceType::Set(rb, 0.0);
SparseSpaceType::Resize(rDx, SparseSpaceType::Size1(rStiffnessMatrix));
SparseSpaceType::Set(rDx, 0.0);
}

KRATOS_INFO_IF("System Construction Time", BaseType::GetEchoLevel() > 0 && rank == 0)
<< system_construction_time.ElapsedSeconds() << std::endl;

pBuilderAndSolver->InitializeSolutionStep(BaseType::GetModelPart(),
rStiffnessMatrix, rDx, rb);

pScheme->InitializeSolutionStep(BaseType::GetModelPart(), rStiffnessMatrix, rDx, rb);

KRATOS_INFO_IF("EigensolverStrategy", BaseType::GetEchoLevel() > 2 && rank == 0)
<<  "Exiting InitializeSolutionStep" << std::endl;

KRATOS_CATCH("")
}

bool SolveSolutionStep() override
{
KRATOS_TRY;

ModelPart& rModelPart = BaseType::GetModelPart();

SchemePointerType& pScheme = this->pGetScheme();
SparseMatrixType& rMassMatrix = this->GetMassMatrix();
SparseMatrixType& rStiffnessMatrix = this->GetStiffnessMatrix();

SparseVectorType b;
SparseSpaceType::Resize(b,SparseSpaceType::Size1(rMassMatrix));
SparseSpaceType::Set(b,0.0);

const bool matrix_contains_dirichlet_dofs = SparseSpaceType::Size1(rMassMatrix) == this->pGetBuilderAndSolver()->GetDofSet().size();

const bool master_slave_constraints_defined = rModelPart.NumberOfMasterSlaveConstraints() != 0;

rModelPart.GetProcessInfo()[BUILD_LEVEL] = 1;
TSparseSpace::SetToZero(rMassMatrix);

EntitiesUtilities::InitializeNonLinearIterationAllEntities(rModelPart);

this->pGetBuilderAndSolver()->Build(pScheme,rModelPart,rMassMatrix,b);
if (master_slave_constraints_defined) {
this->pGetBuilderAndSolver()->ApplyConstraints(pScheme, rModelPart, rMassMatrix, b);
}
if (matrix_contains_dirichlet_dofs) {
this->ApplyDirichletConditions(rMassMatrix, mMassMatrixDiagonalValue);
}

if (BaseType::GetEchoLevel() == 4) {
TSparseSpace::WriteMatrixMarketMatrix("MassMatrix.mm", rMassMatrix, false);
}

rModelPart.GetProcessInfo()[BUILD_LEVEL] = 2;
TSparseSpace::SetToZero(rStiffnessMatrix);
this->pGetBuilderAndSolver()->Build(pScheme,rModelPart,rStiffnessMatrix,b);
if (master_slave_constraints_defined) {
this->pGetBuilderAndSolver()->ApplyConstraints(pScheme, rModelPart, rStiffnessMatrix, b);
}

if (matrix_contains_dirichlet_dofs) {
this->ApplyDirichletConditions(rStiffnessMatrix, mStiffnessMatrixDiagonalValue);
}

EntitiesUtilities::FinalizeNonLinearIterationAllEntities(rModelPart);

if (BaseType::GetEchoLevel() == 4) {
TSparseSpace::WriteMatrixMarketMatrix("StiffnessMatrix.mm", rStiffnessMatrix, false);
}

DenseVectorType Eigenvalues;
DenseMatrixType Eigenvectors;

BuiltinTimer system_solve_time;
this->pGetBuilderAndSolver()->GetLinearSystemSolver()->Solve(
rStiffnessMatrix,
rMassMatrix,
Eigenvalues,
Eigenvectors);

KRATOS_INFO_IF("System Solve Time", BaseType::GetEchoLevel() > 0)
<< system_solve_time.ElapsedSeconds() << std::endl;

if (master_slave_constraints_defined){
this->ReconstructSlaveSolution(Eigenvectors);
}

this->AssignVariables(Eigenvalues,Eigenvectors);


if (mComputeModalDecompostion) {
ComputeModalDecomposition(Eigenvectors);
}

return true;
KRATOS_CATCH("")
}

void FinalizeSolutionStep() override
{
KRATOS_TRY;

const int rank = BaseType::GetModelPart().GetCommunicator().MyPID();
KRATOS_INFO_IF("EigensolverStrategy", BaseType::GetEchoLevel() > 2 && rank == 0)
<<  "Entering FinalizeSolutionStep" << std::endl;

SparseMatrixType& rStiffnessMatrix = this->GetStiffnessMatrix();
SparseVectorPointerType pDx = SparseSpaceType::CreateEmptyVectorPointer();
SparseVectorPointerType pb = SparseSpaceType::CreateEmptyVectorPointer();
pGetBuilderAndSolver()->FinalizeSolutionStep(
BaseType::GetModelPart(), rStiffnessMatrix, *pDx, *pb);
pGetScheme()->FinalizeSolutionStep(BaseType::GetModelPart(),
rStiffnessMatrix, *pDx, *pb);
KRATOS_INFO_IF("EigensolverStrategy", BaseType::GetEchoLevel() > 2 && rank == 0)
<<  "Exiting FinalizeSolutionStep" << std::endl;

KRATOS_CATCH("");
}


int Check() override
{
KRATOS_TRY

ModelPart& rModelPart = BaseType::GetModelPart();
const int rank = rModelPart.GetCommunicator().MyPID();

KRATOS_INFO_IF("EigensolverStrategy", BaseType::GetEchoLevel() > 2 && rank == 0)
<<  "Entering Check" << std::endl;

BaseType::Check();

this->pGetScheme()->Check(rModelPart);

this->pGetBuilderAndSolver()->Check(rModelPart);

KRATOS_INFO_IF("EigensolverStrategy", BaseType::GetEchoLevel() > 2 && rank == 0)
<<  "Exiting Check" << std::endl;

return 0;

KRATOS_CATCH("")
}





protected:








private:


SchemePointerType mpScheme;

BuilderAndSolverPointerType mpBuilderAndSolver;

SparseMatrixPointerType mpMassMatrix;

SparseMatrixPointerType mpStiffnessMatrix;

bool mInitializeWasPerformed = false;

double mMassMatrixDiagonalValue = 0.0;
double mStiffnessMatrixDiagonalValue = 1.0;

bool mComputeModalDecompostion = false;




void ReconstructSlaveSolution(
DenseMatrixType& rEigenvectors
)
{
KRATOS_TRY

auto& r_model_part = BaseType::GetModelPart();
const std::size_t number_of_eigenvalues = rEigenvectors.size1();

struct TLS{
Matrix relation_matrix;
Vector constant_vector;
Vector master_dofs_values;
};

for (std::size_t i_eigenvalue = 0; i_eigenvalue < number_of_eigenvalues; ++i_eigenvalue){

block_for_each(r_model_part.MasterSlaveConstraints(), [&i_eigenvalue, &rEigenvectors](const MasterSlaveConstraint& r_master_slave_constraint){
const auto& r_slave_dofs_vector = r_master_slave_constraint.GetSlaveDofsVector();
for (const auto& r_slave_dof: r_slave_dofs_vector){
AtomicMult(rEigenvectors(i_eigenvalue, r_slave_dof->EquationId()), 0.0);
}
});

block_for_each(r_model_part.MasterSlaveConstraints(), TLS(), [&i_eigenvalue, &rEigenvectors, &r_model_part](const MasterSlaveConstraint& r_master_slave_constraint, TLS& rTLS){
bool constraint_is_active = true;
if (r_master_slave_constraint.IsDefined(ACTIVE))
constraint_is_active = r_master_slave_constraint.Is(ACTIVE);
if (constraint_is_active) {
const auto& r_master_dofs_vector = r_master_slave_constraint.GetMasterDofsVector();
const auto& r_slave_dofs_vector = r_master_slave_constraint.GetSlaveDofsVector();
rTLS.master_dofs_values.resize(r_master_dofs_vector.size());
for (IndexType i = 0; i < r_master_dofs_vector.size(); ++i) {
rTLS.master_dofs_values[i] = rEigenvectors(i_eigenvalue, r_master_dofs_vector[i]->EquationId());
}
r_master_slave_constraint.GetLocalSystem(rTLS.relation_matrix, rTLS.constant_vector, r_model_part.GetProcessInfo());
double aux;
for (IndexType i = 0; i < rTLS.relation_matrix.size1(); ++i) {
aux = rTLS.constant_vector[i];
for(IndexType j = 0; j < rTLS.relation_matrix.size2(); ++j) {
aux += rTLS.relation_matrix(i,j) * rTLS.master_dofs_values[j];
}
AtomicAdd(rEigenvectors(i_eigenvalue, r_slave_dofs_vector[i]->EquationId()), aux);
}
}
});
}

KRATOS_CATCH("")
}


void ApplyDirichletConditions(
SparseMatrixType& rA,
double Factor)
{
KRATOS_TRY

const int rank = BaseType::GetModelPart().GetCommunicator().MyPID();

KRATOS_INFO_IF("EigensolverStrategy", BaseType::GetEchoLevel() > 2 && rank == 0)
<<  "Entering ApplyDirichletConditions" << std::endl;

const std::size_t SystemSize = rA.size1();
std::vector<double> ScalingFactors(SystemSize);
auto& rDofSet = this->pGetBuilderAndSolver()->GetDofSet();
const int NumDofs = static_cast<int>(rDofSet.size());

IndexPartition(NumDofs).for_each([&rDofSet, &ScalingFactors](std::size_t k){
auto dof_iterator = std::begin(rDofSet) + k;
ScalingFactors[k] = (dof_iterator->IsFixed()) ? 0.0 : 1.0;
});

double* AValues = std::begin(rA.value_data());
std::size_t* ARowIndices = std::begin(rA.index1_data());
std::size_t* AColIndices = std::begin(rA.index2_data());


IndexPartition(SystemSize).for_each([&](std::size_t k){
std::size_t ColBegin = ARowIndices[k];
std::size_t ColEnd = ARowIndices[k+1];
if (ScalingFactors[k] == 0.0)
{
for (std::size_t j = ColBegin; j < ColEnd; ++j)
{
if (AColIndices[j] != k)
{
AValues[j] = 0.0;
}
else
{
AValues[j] *= Factor;
}
}
}
else
{
for (std::size_t j = ColBegin; j < ColEnd; ++j)
{
AValues[j] *= ScalingFactors[AColIndices[j]];
}
}
});

KRATOS_INFO_IF("EigensolverStrategy", BaseType::GetEchoLevel() > 2 && rank == 0)
<<  "Exiting ApplyDirichletConditions" << std::endl;

KRATOS_CATCH("")
}

void AssignVariables(DenseVectorType& rEigenvalues, DenseMatrixType& rEigenvectors)
{
ModelPart& rModelPart = BaseType::GetModelPart();
const std::size_t NumEigenvalues = rEigenvalues.size();

rModelPart.GetProcessInfo()[EIGENVALUE_VECTOR] = rEigenvalues;

const auto& r_dof_set = this->pGetBuilderAndSolver()->GetDofSet();

for (ModelPart::NodeIterator itNode = rModelPart.NodesBegin(); itNode!= rModelPart.NodesEnd(); itNode++) {
ModelPart::NodeType::DofsContainerType& NodeDofs = itNode->GetDofs();
const std::size_t NumNodeDofs = NodeDofs.size();
Matrix& rNodeEigenvectors = itNode->GetValue(EIGENVECTOR_MATRIX);
if (rNodeEigenvectors.size1() != NumEigenvalues || rNodeEigenvectors.size2() != NumNodeDofs) {
rNodeEigenvectors.resize(NumEigenvalues,NumNodeDofs,false);
}

for (std::size_t i = 0; i < NumEigenvalues; i++) {
for (std::size_t j = 0; j < NumNodeDofs; j++)
{
const auto itDof = std::begin(NodeDofs) + j;
bool is_active = !(r_dof_set.find(**itDof) == r_dof_set.end());
if ((*itDof)->IsFree() && is_active) {
rNodeEigenvectors(i,j) = rEigenvectors(i,(*itDof)->EquationId());
}
else {
rNodeEigenvectors(i,j) = 0.0;
}
}
}
}
}


void ComputeModalDecomposition(const DenseMatrixType& rEigenvectors)
{
const SparseMatrixType& rMassMatrix = this->GetMassMatrix();
SparseMatrixType m_temp = ZeroMatrix(rEigenvectors.size1(),rEigenvectors.size2());
boost::numeric::ublas::axpy_prod(rEigenvectors,rMassMatrix,m_temp,true);
Matrix modal_mass_matrix = ZeroMatrix(m_temp.size1(),m_temp.size1());
boost::numeric::ublas::axpy_prod(m_temp,trans(rEigenvectors),modal_mass_matrix);

const SparseMatrixType& rStiffnessMatrix = this->GetStiffnessMatrix();
SparseMatrixType k_temp = ZeroMatrix(rEigenvectors.size1(),rEigenvectors.size2());
boost::numeric::ublas::axpy_prod(rEigenvectors,rStiffnessMatrix,k_temp,true);
Matrix modal_stiffness_matrix = ZeroMatrix(k_temp.size1(),k_temp.size1());
boost::numeric::ublas::axpy_prod(k_temp,trans(rEigenvectors),modal_stiffness_matrix);

ModelPart& rModelPart = BaseType::GetModelPart();
rModelPart.GetProcessInfo()[MODAL_MASS_MATRIX] = modal_mass_matrix;
rModelPart.GetProcessInfo()[MODAL_STIFFNESS_MATRIX] = modal_stiffness_matrix;

KRATOS_INFO("ModalMassMatrix")      << modal_mass_matrix << std::endl;
KRATOS_INFO("ModalStiffnessMatrix") << modal_stiffness_matrix << std::endl;
}




}; 





} 
