
#pragma once



#include "solving_strategies/strategies/implicit_solving_strategy.h"
#include "utilities/builtin_timer.h"

#include "structural_mechanics_application_variables.h"

namespace Kratos
{







template <class TSparseSpace,
class TDenseSpace,
class TLinearSolver>
class PrebucklingStrategy
: public ImplicitSolvingStrategy<TSparseSpace, TDenseSpace, TLinearSolver>
{
public:

KRATOS_CLASS_POINTER_DEFINITION(PrebucklingStrategy);

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

typedef ConvergenceCriteria<TSparseSpace, TDenseSpace> ConvergenceCriteriaType;



PrebucklingStrategy(
ModelPart &rModelPart,
SchemePointerType pScheme,
BuilderAndSolverPointerType pEigenSolver,
BuilderAndSolverPointerType pBuilderAndSolver,
typename ConvergenceCriteriaType::Pointer pConvergenceCriteria,
int MaxIteration,
Parameters BucklingSettings )
: ImplicitSolvingStrategy<TSparseSpace, TDenseSpace, TLinearSolver>(rModelPart)
{
KRATOS_TRY

mpScheme = pScheme;

mpEigenSolver = pEigenSolver;

mpBuilderAndSolver = pBuilderAndSolver;

mpConvergenceCriteria = pConvergenceCriteria;

mMaxIteration = MaxIteration;

mInitialLoadIncrement = BucklingSettings["initial_load_increment"].GetDouble();

mSmallLoadIncrement = BucklingSettings["small_load_increment"].GetDouble();

mPathFollowingStep = BucklingSettings["path_following_step"].GetDouble();

mConvergenceRatio = BucklingSettings["convergence_ratio"].GetDouble();

mMakeMatricesSymmetricFlag = BucklingSettings["make_matrices_symmetric"].GetBool();

mpEigenSolver->SetDofSetIsInitializedFlag(false);
mpEigenSolver->SetReshapeMatrixFlag(false);
mpEigenSolver->SetCalculateReactionsFlag(false);
mpBuilderAndSolver->SetDofSetIsInitializedFlag(false);
mpBuilderAndSolver->SetCalculateReactionsFlag(false);
mpBuilderAndSolver->SetReshapeMatrixFlag(false);

this->SetEchoLevel(1);

this->SetRebuildLevel(1);

mpStiffnessMatrix = TSparseSpace::CreateEmptyMatrixPointer();
mpStiffnessMatrixPrevious = TSparseSpace::CreateEmptyMatrixPointer();
mpDx = TSparseSpace::CreateEmptyVectorPointer();
mpRHS = TSparseSpace::CreateEmptyVectorPointer();

rModelPart.GetProcessInfo()[TIME] = 1.0;

KRATOS_CATCH("")
}

PrebucklingStrategy(const PrebucklingStrategy &Other) = delete;


~PrebucklingStrategy() override
{
this->Clear();
}




SchemePointerType &pGetScheme()
{
return mpScheme;
};


BuilderAndSolverPointerType &pGetEigenSolver()
{
return mpEigenSolver;
};


BuilderAndSolverPointerType &pGetBuilderAndSolver()
{
return mpBuilderAndSolver;
};


ConvergenceCriteriaType &GetConvergenceCriteria()
{
return mpConvergenceCriteria;
}


bool GetSolutionFoundFlag()
{
return mSolutionFound;
}


void SetEchoLevel(int Level) override
{
BaseType::SetEchoLevel(Level);
this->pGetEigenSolver()->SetEchoLevel(Level);
this->pGetBuilderAndSolver()->SetEchoLevel(Level);
}


void Initialize() override
{
KRATOS_TRY

ModelPart &rModelPart = BaseType::GetModelPart();

KRATOS_INFO_IF("PrebucklingStrategy", BaseType::GetEchoLevel() > 2 )
<< "Entering Initialize" << std::endl;

if (!mInitializeWasPerformed)
{
SchemePointerType &pScheme = this->pGetScheme();

if ( !pScheme->SchemeIsInitialized() )
pScheme->Initialize(rModelPart);

if ( !pScheme->ElementsAreInitialized() )
pScheme->InitializeElements(rModelPart);

if ( !pScheme->ConditionsAreInitialized() )
pScheme->InitializeConditions(rModelPart);
}
mpConvergenceCriteria->Initialize(BaseType::GetModelPart());

mInitializeWasPerformed = true;

KRATOS_INFO_IF("PrebucklingStrategy", BaseType::GetEchoLevel() > 2 )
<< "Exiting Initialize" << std::endl;

KRATOS_CATCH("")
}


void Clear() override
{
KRATOS_TRY

BuilderAndSolverPointerType &pBuilderAndSolver = this->pGetBuilderAndSolver();
pBuilderAndSolver->GetLinearSystemSolver()->Clear();
BuilderAndSolverPointerType &pEigenSolver = this->pGetEigenSolver();
pEigenSolver->GetLinearSystemSolver()->Clear();

if (mpStiffnessMatrix != nullptr)
mpStiffnessMatrix = nullptr;

if (mpStiffnessMatrixPrevious != nullptr)
mpStiffnessMatrixPrevious = nullptr;

if (mpRHS != nullptr)
mpRHS = nullptr;

if (mpDx != nullptr)
mpDx = nullptr;

pBuilderAndSolver->SetDofSetIsInitializedFlag(false);
pEigenSolver->SetDofSetIsInitializedFlag(false);

pBuilderAndSolver->Clear();
pEigenSolver->Clear();

this->pGetScheme()->Clear();

mInitializeWasPerformed = false;
mSolutionStepIsInitialized = false;


KRATOS_CATCH("")
}


void InitializeSolutionStep() override
{
KRATOS_TRY
if (!mSolutionStepIsInitialized){
ModelPart &rModelPart = BaseType::GetModelPart();

KRATOS_INFO_IF("PrebucklingStrategy", BaseType::GetEchoLevel() > 2 )
<< "Entering InitializeSolutionStep" << std::endl;

BuilderAndSolverPointerType &pBuilderAndSolver = this->pGetBuilderAndSolver();
SchemePointerType &pScheme = this->pGetScheme();
typename ConvergenceCriteriaType::Pointer pConvergenceCriteria = mpConvergenceCriteria;
SparseMatrixType& rStiffnessMatrix  = *mpStiffnessMatrix;
SparseVectorType& rRHS  = *mpRHS;
SparseVectorType& rDx  = *mpDx;

SparseVectorPointerType _pDx = SparseSpaceType::CreateEmptyVectorPointer();
SparseVectorPointerType _pb = SparseSpaceType::CreateEmptyVectorPointer();

BuiltinTimer system_construction_time;
if ( !pBuilderAndSolver->GetDofSetIsInitializedFlag() ||
pBuilderAndSolver->GetReshapeMatrixFlag() )
{
BuiltinTimer setup_dofs_time;

pBuilderAndSolver->SetUpDofSet(pScheme, rModelPart);

KRATOS_INFO_IF("Setup Dofs Time", BaseType::GetEchoLevel() > 0 )
<< setup_dofs_time.ElapsedSeconds() << std::endl;

BuiltinTimer setup_system_time;

pBuilderAndSolver->SetUpSystem(rModelPart);

KRATOS_INFO_IF("Setup System Time", BaseType::GetEchoLevel() > 0 )
<< setup_system_time.ElapsedSeconds() << std::endl;

BuiltinTimer system_matrix_resize_time;

pBuilderAndSolver->ResizeAndInitializeVectors(
pScheme, mpStiffnessMatrix, mpDx, mpRHS, rModelPart);
pBuilderAndSolver->ResizeAndInitializeVectors(
pScheme, mpStiffnessMatrixPrevious, _pDx, _pb, rModelPart);

KRATOS_INFO_IF("System Matrix Resize Time", BaseType::GetEchoLevel() > 0 )
<< system_matrix_resize_time.ElapsedSeconds() << std::endl;

}

KRATOS_INFO_IF("System Construction Time", BaseType::GetEchoLevel() > 0 )
<< system_construction_time.ElapsedSeconds() << std::endl;

pBuilderAndSolver->InitializeSolutionStep(BaseType::GetModelPart(),
rStiffnessMatrix, rDx, rRHS);

pScheme->InitializeSolutionStep(BaseType::GetModelPart(), rStiffnessMatrix, rDx, rRHS);

pConvergenceCriteria->InitializeSolutionStep(BaseType::GetModelPart(), pBuilderAndSolver->GetDofSet(), rStiffnessMatrix, rDx, rRHS);

mSolutionStepIsInitialized = true;

KRATOS_INFO_IF("PrebucklingStrategy", BaseType::GetEchoLevel() > 2 )
<< "Exiting InitializeSolutionStep" << std::endl;
}
KRATOS_CATCH("")
}


bool SolveSolutionStep() override
{
KRATOS_TRY

ModelPart& rModelPart = BaseType::GetModelPart();
SchemePointerType& pScheme = this->pGetScheme();
BuilderAndSolverPointerType& pBuilderAndSolver = this->pGetBuilderAndSolver();
SparseMatrixType& rStiffnessMatrix  = *mpStiffnessMatrix;
SparseMatrixType& rStiffnessMatrixPrevious = *mpStiffnessMatrixPrevious;
SparseVectorType& rRHS  = *mpRHS;
SparseVectorType& rDx  = *mpDx;

typename ConvergenceCriteriaType::Pointer pConvergenceCriteria = mpConvergenceCriteria;

unsigned int iteration_number = 1;
rModelPart.GetProcessInfo()[NL_ITERATION_NUMBER] = iteration_number;
bool is_converged = false;

double delta_load_multiplier = 0.0;
if( mLoadStepIteration == 1) 
{
delta_load_multiplier = mInitialLoadIncrement*(mLambda + mLambdaPrev);
}
else if( mLoadStepIteration % 2 == 1 ) 
{
delta_load_multiplier = mSmallLoadIncrement*(mLambdaPrev );
}

BuiltinTimer system_solve_time;
this->pGetScheme()->InitializeNonLinIteration( rModelPart,rStiffnessMatrix, rDx, rRHS );
pConvergenceCriteria->InitializeNonLinearIteration(rModelPart, pBuilderAndSolver->GetDofSet(), rStiffnessMatrix, rDx, rRHS);
is_converged = mpConvergenceCriteria->PreCriteria(rModelPart, pBuilderAndSolver->GetDofSet(), rStiffnessMatrix, rDx, rRHS);

TSparseSpace::SetToZero(rStiffnessMatrix);
TSparseSpace::SetToZero(rRHS);
TSparseSpace::SetToZero(rDx);
pBuilderAndSolver->BuildAndSolve(pScheme, rModelPart, rStiffnessMatrix, rDx, rRHS);
pScheme->Update(rModelPart, pBuilderAndSolver->GetDofSet(), rStiffnessMatrix, rDx, rRHS );
BaseType::MoveMesh();
pScheme->FinalizeNonLinIteration( rModelPart,rStiffnessMatrix, rDx, rRHS );
pConvergenceCriteria->FinalizeNonLinearIteration(rModelPart, pBuilderAndSolver->GetDofSet(), rStiffnessMatrix, rDx, rRHS);

if (is_converged){
is_converged = mpConvergenceCriteria->PostCriteria(rModelPart, pBuilderAndSolver->GetDofSet(), rStiffnessMatrix, rDx, rRHS);
}

while ( !is_converged &&
iteration_number++ < mMaxIteration)
{
rModelPart.GetProcessInfo()[NL_ITERATION_NUMBER] = iteration_number;

pScheme->InitializeNonLinIteration( rModelPart,rStiffnessMatrix, rDx, rRHS );
pConvergenceCriteria->InitializeNonLinearIteration(rModelPart, pBuilderAndSolver->GetDofSet(), rStiffnessMatrix, rDx, rRHS);
is_converged = mpConvergenceCriteria->PreCriteria(rModelPart, pBuilderAndSolver->GetDofSet(), rStiffnessMatrix, rDx, rRHS);

TSparseSpace::SetToZero(rStiffnessMatrix);
TSparseSpace::SetToZero(rDx);
TSparseSpace::SetToZero(rRHS);
pBuilderAndSolver->BuildAndSolve(pScheme, rModelPart, rStiffnessMatrix,rDx, rRHS);

this->pGetScheme()->Update(rModelPart, pBuilderAndSolver->GetDofSet(), rStiffnessMatrix, rDx, rRHS);
BaseType::MoveMesh();

this->pGetScheme()->FinalizeNonLinIteration( rModelPart,rStiffnessMatrix, rDx, rRHS );
pConvergenceCriteria->FinalizeNonLinearIteration(rModelPart, pBuilderAndSolver->GetDofSet(), rStiffnessMatrix, rDx, rRHS);

if (is_converged){
is_converged = mpConvergenceCriteria->PostCriteria(rModelPart, pBuilderAndSolver->GetDofSet(), rStiffnessMatrix, rDx, rRHS);
}
}
KRATOS_INFO_IF("Nonlinear Loadstep Time: ", BaseType::GetEchoLevel() > 0)
<< system_solve_time.ElapsedSeconds() << std::endl;

if ( !is_converged ) {
KRATOS_INFO_IF("Nonlinear Loadstep: ", this->GetEchoLevel() > 0)
<< "Convergence not achieved after ( " << mMaxIteration
<< " ) Iterations !" << std::endl;
} else {
KRATOS_INFO_IF("Nonlinear Loadstep: ", this->GetEchoLevel() > 0)
<< "Convergence achieved after " << iteration_number << " / "
<< mMaxIteration << " iterations" << std::endl;
}

DenseVectorType Eigenvalues;
DenseMatrixType Eigenvectors;

if( mLoadStepIteration % 2 == 0 ){
rStiffnessMatrixPrevious = rStiffnessMatrix;

if( mLoadStepIteration > 0){
mLambdaPrev = mLambdaPrev + mPathFollowingStep*mLambda;
}
}
else if( mLoadStepIteration % 2 == 1 ){



rStiffnessMatrix = rStiffnessMatrixPrevious - rStiffnessMatrix;

if( mMakeMatricesSymmetricFlag ){
rStiffnessMatrix = 0.5 * ( rStiffnessMatrix + boost::numeric::ublas::trans(rStiffnessMatrix) );
rStiffnessMatrixPrevious = 0.5 * ( rStiffnessMatrixPrevious + boost::numeric::ublas::trans(rStiffnessMatrixPrevious) );
}

this->pGetEigenSolver()->GetLinearSystemSolver()->Solve(
rStiffnessMatrixPrevious,
rStiffnessMatrix,
Eigenvalues,
Eigenvectors);

mLambda = Eigenvalues(0)*delta_load_multiplier;
for(unsigned int i = 0; i < Eigenvalues.size(); i++ )
{
Eigenvalues[i] = mLambdaPrev + Eigenvalues[i]*delta_load_multiplier;
}

this->AssignVariables(Eigenvalues, Eigenvectors);

mpStiffnessMatrix = nullptr;
pBuilderAndSolver->ResizeAndInitializeVectors(
pScheme, mpStiffnessMatrix, mpDx, mpRHS, rModelPart);

if( std::abs(mLambda/mLambdaPrev) < mConvergenceRatio ){
mSolutionFound = true;
KRATOS_INFO_IF("Prebuckling Analysis: ", BaseType::GetEchoLevel() > 0)
<< "Convergence achieved in " << mLoadStepIteration + 1 << " Load Iterations!" << std::endl;
}
}

mLoadStepIteration++;
UpdateLoadConditions();

return true;
KRATOS_CATCH("")
}


void FinalizeSolutionStep() override
{
KRATOS_TRY;

KRATOS_INFO_IF("PrebucklingStrategy", BaseType::GetEchoLevel() > 2 )
<< "Entering FinalizeSolutionStep" << std::endl;

typename ConvergenceCriteriaType::Pointer pConvergenceCriteria = mpConvergenceCriteria;
SparseMatrixType& rStiffnessMatrix  = *mpStiffnessMatrix;
SparseVectorType& rRHS  = *mpRHS;
SparseVectorType& rDx  = *mpDx;

pGetBuilderAndSolver()->FinalizeSolutionStep(
BaseType::GetModelPart(), rStiffnessMatrix, rDx, rRHS);

pGetScheme()->FinalizeSolutionStep(BaseType::GetModelPart(),
rStiffnessMatrix, rDx, rRHS);

pConvergenceCriteria->FinalizeSolutionStep(BaseType::GetModelPart(), pGetBuilderAndSolver()->GetDofSet(), rStiffnessMatrix, rDx, rRHS );

pGetScheme()->Clean();
mSolutionStepIsInitialized = false;

KRATOS_INFO_IF("PrebucklingStrategy", BaseType::GetEchoLevel() > 2 )
<< "Exiting FinalizeSolutionStep" << std::endl;

KRATOS_CATCH("");
}


int Check() override
{
KRATOS_TRY

ModelPart &rModelPart = BaseType::GetModelPart();

KRATOS_INFO_IF("PrebucklingStrategy", BaseType::GetEchoLevel() > 2 )
<< "Entering Check" << std::endl;

BaseType::Check();

this->pGetScheme()->Check(rModelPart);

this->pGetBuilderAndSolver()->Check(rModelPart);

KRATOS_INFO_IF("PrebucklingStrategy", BaseType::GetEchoLevel() > 2 )
<< "Exiting Check" << std::endl;

return 0;

KRATOS_CATCH("")
}





protected:








private:


SchemePointerType mpScheme;

BuilderAndSolverPointerType mpEigenSolver;

BuilderAndSolverPointerType mpBuilderAndSolver;


SparseMatrixPointerType mpStiffnessMatrix;

SparseMatrixPointerType mpStiffnessMatrixPrevious;

SparseVectorPointerType mpRHS;

SparseVectorPointerType mpDx;

typename ConvergenceCriteriaType::Pointer mpConvergenceCriteria;

bool mInitializeWasPerformed = false;

bool mSolutionStepIsInitialized = false;

bool mSolutionFound = false;

unsigned int mLoadStepIteration = 0;

unsigned int mMaxIteration;

std::vector<array_1d<double,3>> mpInitialLoads;

double mInitialLoadIncrement;
double mSmallLoadIncrement;
double mPathFollowingStep;
double mConvergenceRatio;

double mLambda = 0.0;
double mLambdaPrev = 1.0;

bool mMakeMatricesSymmetricFlag;




void UpdateLoadConditions()
{
ModelPart &rModelPart = BaseType::GetModelPart();

if( mLoadStepIteration == 1){
rModelPart.GetProcessInfo()[TIME] = ( 1.0 + mInitialLoadIncrement );
}
else if( mLoadStepIteration % 2 == 0){
rModelPart.GetProcessInfo()[TIME] = ( mLambdaPrev + mPathFollowingStep * mLambda );
}
else{
rModelPart.GetProcessInfo()[TIME] = (1 + mSmallLoadIncrement) * mLambdaPrev;
}
}


void AssignVariables(DenseVectorType &rEigenvalues, DenseMatrixType &rEigenvectors )
{
ModelPart &rModelPart = BaseType::GetModelPart();
const std::size_t NumEigenvalues = rEigenvalues.size();

rModelPart.GetProcessInfo()[EIGENVALUE_VECTOR] = rEigenvalues;

for (ModelPart::NodeIterator itNode = rModelPart.NodesBegin(); itNode != rModelPart.NodesEnd(); itNode++)
{
ModelPart::NodeType::DofsContainerType &NodeDofs = itNode->GetDofs();
const std::size_t NumNodeDofs = NodeDofs.size();

Matrix &rNodeEigenvectors = itNode->GetValue(EIGENVECTOR_MATRIX);
if (rNodeEigenvectors.size1() != NumEigenvalues || rNodeEigenvectors.size2() != NumNodeDofs){
rNodeEigenvectors.resize(NumEigenvalues, NumNodeDofs, false);
}


for (std::size_t i = 0; i < NumEigenvalues; i++){
for (std::size_t j = 0; j < NumNodeDofs; j++){
auto itDof = std::begin(NodeDofs) + j;
if( !(*itDof)->IsFixed() ){
rNodeEigenvectors(i, j) = rEigenvectors(i, (*itDof)->EquationId());
}
else{
rNodeEigenvectors(i, j) = 0.0;
}
}
}
}
}




}; 




} 
