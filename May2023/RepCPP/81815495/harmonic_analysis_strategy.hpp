
#pragma once



#include "solving_strategies/strategies/implicit_solving_strategy.h"
#include "utilities/builtin_timer.h"

#include "structural_mechanics_application_variables.h"

namespace Kratos
{






template<class TSparseSpace,
class TDenseSpace,
class TLinearSolver
>
class HarmonicAnalysisStrategy
: public ImplicitSolvingStrategy<TSparseSpace, TDenseSpace, TLinearSolver>
{
public:

KRATOS_CLASS_POINTER_DEFINITION(HarmonicAnalysisStrategy);

typedef ImplicitSolvingStrategy<TSparseSpace, TDenseSpace, TLinearSolver> BaseType;

typedef typename BaseType::TSchemeType::Pointer SchemePointerType;

typedef typename BaseType::TBuilderAndSolverType::Pointer BuilderAndSolverPointerType;

typedef TDenseSpace DenseSpaceType;

typedef typename TDenseSpace::VectorType DenseVectorType;

typedef typename TDenseSpace::MatrixType DenseMatrixType;

typedef typename TDenseSpace::MatrixPointerType DenseMatrixPointerType;

typedef TSparseSpace SparseSpaceType;

typedef typename TSparseSpace::VectorPointerType SparseVectorPointerType;

typedef std::complex<double> ComplexType;

typedef DenseVector<ComplexType> ComplexVectorType;


HarmonicAnalysisStrategy(
ModelPart& rModelPart,
SchemePointerType pScheme,
BuilderAndSolverPointerType pBuilderAndSolver,
bool UseMaterialDampingFlag = false
)
: ImplicitSolvingStrategy<TSparseSpace, TDenseSpace, TLinearSolver>(rModelPart)
{
KRATOS_TRY

mpScheme = pScheme;

mpBuilderAndSolver = pBuilderAndSolver;

mpBuilderAndSolver->SetDofSetIsInitializedFlag(false);

mpForceVector = SparseSpaceType::CreateEmptyVectorPointer();
mpModalMatrix = DenseSpaceType::CreateEmptyMatrixPointer();

this->SetUseMaterialDampingFlag(UseMaterialDampingFlag);

this->SetEchoLevel(0);

this->SetRebuildLevel(0);

KRATOS_CATCH("")
}

HarmonicAnalysisStrategy(const HarmonicAnalysisStrategy& Other) = delete;

~HarmonicAnalysisStrategy() override
{
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
}

SchemePointerType& pGetScheme()
{
return mpScheme;
}

void SetBuilderAndSolver(BuilderAndSolverPointerType pNewBuilderAndSolver)
{
mpBuilderAndSolver = pNewBuilderAndSolver;
}

BuilderAndSolverPointerType& pGetBuilderAndSolver()
{
return mpBuilderAndSolver;
}

void SetReformDofSetAtEachStepFlag(bool flag)
{
this->pGetBuilderAndSolver()->SetReshapeMatrixFlag(flag);
}

bool GetReformDofSetAtEachStepFlag() const
{
return this->pGetBuilderAndSolver()->GetReshapeMatrixFlag();
}

void SetUseMaterialDampingFlag(bool flag)
{
mUseMaterialDamping = flag;
}

bool GetUseMaterialDampingFlag() const
{
return mUseMaterialDamping;
}


void SetEchoLevel(int Level) override
{
BaseType::SetEchoLevel(Level);
this->pGetBuilderAndSolver()->SetEchoLevel(Level);
}

void Initialize() override
{
KRATOS_TRY

auto& r_model_part = BaseType::GetModelPart();
const auto rank = r_model_part.GetCommunicator().MyPID();

KRATOS_INFO_IF("HarmonicAnalysisStrategy", BaseType::GetEchoLevel() > 2 && rank == 0)
<<  "Entering Initialize" << std::endl;

if( !mInitializeWasPerformed )
{
auto& r_process_info = r_model_part.GetProcessInfo();

auto& p_scheme = this->pGetScheme();

if (p_scheme->SchemeIsInitialized() == false)
p_scheme->Initialize(r_model_part);

if (p_scheme->ElementsAreInitialized() == false)
p_scheme->InitializeElements(r_model_part);

if (p_scheme->ConditionsAreInitialized() == false)
p_scheme->InitializeConditions(r_model_part);

auto& p_builder_and_solver = this->pGetBuilderAndSolver();

BuiltinTimer setup_dofs_time;
p_builder_and_solver->SetUpDofSet(p_scheme, r_model_part);
KRATOS_INFO_IF("Setup Dofs Time", BaseType::GetEchoLevel() > 0 && rank == 0)
<< setup_dofs_time.ElapsedSeconds() << std::endl;

BuiltinTimer setup_system_time;
p_builder_and_solver->SetUpSystem(r_model_part);
KRATOS_INFO_IF("Setup System Time", BaseType::GetEchoLevel() > 0 && rank == 0)
<< setup_system_time.ElapsedSeconds() << std::endl;

auto& r_force_vector = *mpForceVector;
const unsigned int system_size = p_builder_and_solver->GetEquationSystemSize();

BuiltinTimer force_vector_build_time;
if (r_force_vector.size() != system_size)
r_force_vector.resize(system_size, false);
r_force_vector = ZeroVector( system_size );
p_builder_and_solver->BuildRHS(p_scheme,r_model_part,r_force_vector);

KRATOS_INFO_IF("Force Vector Build Time", BaseType::GetEchoLevel() > 0 && rank == 0)
<< force_vector_build_time.ElapsedSeconds() << std::endl;

auto& r_modal_matrix = *mpModalMatrix;
const std::size_t n_modes = r_process_info[EIGENVALUE_VECTOR].size();
if( r_modal_matrix.size1() != system_size || r_modal_matrix.size2() != n_modes )
r_modal_matrix.resize( system_size, n_modes, false );
r_modal_matrix = ZeroMatrix( system_size, n_modes );

BuiltinTimer modal_matrix_build_time;
for( std::size_t i = 0; i < n_modes; ++i )
{
for( auto& node : r_model_part.Nodes() )
{
ModelPart::NodeType::DofsContainerType& node_dofs = node.GetDofs();
const std::size_t n_node_dofs = node_dofs.size();
const Matrix& r_node_eigenvectors = node.GetValue(EIGENVECTOR_MATRIX);


for( std::size_t j = 0; j < n_node_dofs; ++j )
{
const auto it_dof = std::begin(node_dofs) + j;
r_modal_matrix((*it_dof)->EquationId(), i) = r_node_eigenvectors(i, j);
}
}
}

KRATOS_INFO_IF("Modal Matrix Build Time", BaseType::GetEchoLevel() > 0 && rank == 0)
<< modal_matrix_build_time.ElapsedSeconds() << std::endl;

for( auto& property : r_model_part.PropertiesArray() )
{
if( property->Has(SYSTEM_DAMPING_RATIO) )
{
mSystemDamping = property->GetValue(SYSTEM_DAMPING_RATIO);
}

if( property->Has(RAYLEIGH_ALPHA) && property->Has(RAYLEIGH_BETA) )
{
mRayleighAlpha = property->GetValue(RAYLEIGH_ALPHA);
mRayleighBeta = property->GetValue(RAYLEIGH_BETA);
}
}

if( mUseMaterialDamping )
{
KRATOS_ERROR_IF(r_model_part.NumberOfSubModelParts() < 1) << "No submodelparts detected!" << std::endl;

r_model_part.GetProcessInfo()[BUILD_LEVEL] = 2;
mMaterialDampingRatios = ZeroVector( n_modes );

auto pDx = SparseSpaceType::CreateEmptyVectorPointer();
auto pb = SparseSpaceType::CreateEmptyVectorPointer();
auto& rDx = *pDx;
auto& rb = *pb;
SparseSpaceType::Resize(rDx,system_size);
SparseSpaceType::Set(rDx,0.0);
SparseSpaceType::Resize(rb,system_size);
SparseSpaceType::Set(rb,0.0);

BuiltinTimer material_damping_build_time;

for( std::size_t i = 0; i < n_modes; ++i )
{
double up = 0.0;
double down = 0.0;
auto modal_vector = column( r_modal_matrix, i );
for( auto& sub_model_part : r_model_part.SubModelParts() )
{
double damping_coefficient = 0.0;
for( auto& property : sub_model_part.PropertiesArray() )
{
if( property->Has(SYSTEM_DAMPING_RATIO) )
{
damping_coefficient = property->GetValue(SYSTEM_DAMPING_RATIO);
}
}

auto temp_stiffness_matrix = SparseSpaceType::CreateEmptyMatrixPointer();
p_builder_and_solver->ResizeAndInitializeVectors(p_scheme,
temp_stiffness_matrix,
pDx,
pb,
r_model_part);

p_builder_and_solver->BuildLHS(p_scheme, sub_model_part, *temp_stiffness_matrix);

double strain_energy = 0.5 * inner_prod( prod(modal_vector, *temp_stiffness_matrix), modal_vector );
down += strain_energy;
up += damping_coefficient * strain_energy;
}
KRATOS_ERROR_IF(down < std::numeric_limits<double>::epsilon()) << "No valid effective "
<< "material damping ratio could be computed. Are all elements to be damped available "
<< "in the submodelparts? Are the modal vectors available?" << std::endl;

mMaterialDampingRatios(i) = up / down;
}

KRATOS_INFO_IF("Material Damping Build Time", BaseType::GetEchoLevel() > 0 && rank == 0)
<< material_damping_build_time.ElapsedSeconds() << std::endl;
}
mInitializeWasPerformed = true;
}

KRATOS_INFO_IF("HarmonicAnalysisStrategy", BaseType::GetEchoLevel() > 2 && rank == 0)
<<  "Exiting Initialize" << std::endl;

KRATOS_CATCH("")
}

bool SolveSolutionStep() override
{
KRATOS_TRY

auto& r_model_part = BaseType::GetModelPart();
const auto rank = r_model_part.GetCommunicator().MyPID();

KRATOS_INFO_IF("HarmonicAnalysisStrategy", BaseType::GetEchoLevel() > 2 && rank == 0)
<<  "Entering SolveSolutionStep" << std::endl;

auto& r_process_info = r_model_part.GetProcessInfo();
double excitation_frequency = r_process_info[TIME];

DenseVectorType eigenvalues = r_process_info[EIGENVALUE_VECTOR];
const std::size_t n_modes = eigenvalues.size();

const std::size_t n_dofs = this->pGetBuilderAndSolver()->GetEquationSystemSize();

auto& f = *mpForceVector;

ComplexType mode_weight;
ComplexVectorType modal_displacement;
modal_displacement.resize(n_dofs, false);
modal_displacement = ZeroVector( n_dofs );

double modal_damping = 0.0;

for( std::size_t i = 0; i < n_modes; ++i )
{
KRATOS_ERROR_IF( eigenvalues[i] < std::numeric_limits<double>::epsilon() ) << "No valid eigenvalue "
<< "for mode " << i << std::endl;
modal_damping = mSystemDamping + mRayleighAlpha / (2 * std::sqrt(eigenvalues[i])) + mRayleighBeta * std::sqrt(eigenvalues[i]) / 2;

if( mUseMaterialDamping )
{
modal_damping += mMaterialDampingRatios[i];
}

auto& r_modal_matrix = *mpModalMatrix;

DenseVectorType modal_vector(n_dofs);
TDenseSpace::GetColumn(i, r_modal_matrix, modal_vector);

ComplexType factor( eigenvalues[i] - std::pow( excitation_frequency, 2.0 ), 2 * modal_damping * std::sqrt(eigenvalues[i]) * excitation_frequency );
KRATOS_ERROR_IF( std::abs(factor) < std::numeric_limits<double>::epsilon() ) << "No valid modal weight" << std::endl;
mode_weight = inner_prod( modal_vector, f ) / factor;

for( auto& node : r_model_part.Nodes() )
{
auto& node_dofs = node.GetDofs();
const std::size_t n_node_dofs = node_dofs.size();
const Matrix& r_node_eigenvectors = node.GetValue(EIGENVECTOR_MATRIX);


for (std::size_t j = 0; j < n_node_dofs; j++)
{
auto it_dof = std::begin(node_dofs) + j;
modal_displacement[(*it_dof)->EquationId()] = modal_displacement[(*it_dof)->EquationId()] + mode_weight * r_node_eigenvectors(i,j);
}
}
}

this->AssignVariables(modal_displacement);

KRATOS_INFO_IF("HarmonicAnalysisStrategy", BaseType::GetEchoLevel() > 2 && rank == 0)
<< "Exiting SolveSolutionStep" << std::endl;

return true;

KRATOS_CATCH("")
}

void Clear() override
{
KRATOS_TRY

auto& p_builder_and_solver = this->pGetBuilderAndSolver();
p_builder_and_solver->GetLinearSystemSolver()->Clear();

SparseSpaceType::Clear(mpForceVector);
DenseSpaceType::Clear(mpModalMatrix);

p_builder_and_solver->SetDofSetIsInitializedFlag(false);

p_builder_and_solver->Clear();

this->pGetScheme()->Clear();

mInitializeWasPerformed = false;
mUseMaterialDamping = false;
mRayleighAlpha = 0.0;
mRayleighBeta = 0.0;
mSystemDamping = 0.0;

KRATOS_CATCH("")
}

int Check() override
{
KRATOS_TRY

auto& r_model_part = BaseType::GetModelPart();
const auto rank = r_model_part.GetCommunicator().MyPID();

KRATOS_INFO_IF("HarmonicAnalysisStrategy", BaseType::GetEchoLevel() > 2 && rank == 0)
<< "Entering Check" << std::endl;

BaseType::Check();

this->pGetScheme()->Check(r_model_part);

this->pGetBuilderAndSolver()->Check(r_model_part);

KRATOS_INFO_IF("HarmonicAnalysisStrategy", BaseType::GetEchoLevel() > 2 && rank == 0)
<< "Exiting Check" << std::endl;

return 0;

KRATOS_CATCH("")
}





protected:








private:


SchemePointerType mpScheme;

BuilderAndSolverPointerType mpBuilderAndSolver;

bool mInitializeWasPerformed = false;

SparseVectorPointerType mpForceVector;

DenseMatrixPointerType mpModalMatrix;

double mRayleighAlpha = 0.0;

double mRayleighBeta = 0.0;

double mSystemDamping = 0.0;

bool mUseMaterialDamping;

vector< double > mMaterialDampingRatios;



void AssignVariables(ComplexVectorType& rModalDisplacement, int step=0)
{
auto& r_model_part = BaseType::GetModelPart();
for( auto& node : r_model_part.Nodes() )
{
ModelPart::NodeType::DofsContainerType& rNodeDofs = node.GetDofs();

for( auto it_dof = std::begin(rNodeDofs); it_dof != std::end(rNodeDofs); it_dof++ )
{
auto& p_dof = *it_dof;
if( !p_dof->IsFixed() )
{
const auto modal_displacement = rModalDisplacement( p_dof->EquationId() );
if( std::real( modal_displacement ) < 0 )
{
p_dof->GetSolutionStepValue(step) = -1 * std::abs( modal_displacement );
}
else
{
p_dof->GetSolutionStepValue(step) = std::abs( modal_displacement );
}

p_dof->GetSolutionStepReactionValue(step) = std::arg( modal_displacement );
}
else
{
p_dof->GetSolutionStepValue(step) = 0.0;
p_dof->GetSolutionStepReactionValue(step) = 0.0;
}
}
}
}




}; 





} 

