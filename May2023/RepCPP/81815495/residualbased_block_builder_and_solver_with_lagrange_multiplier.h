#if !defined(KRATOS_RESIDUAL_BASED_BLOCK_BUILDER_AND_SOLVER_WITH_LAGRANGE_MULTIPLIER )
#define  KRATOS_RESIDUAL_BASED_BLOCK_BUILDER_AND_SOLVER_WITH_LAGRANGE_MULTIPLIER






#include "solving_strategies/builder_and_solvers/residualbased_block_builder_and_solver.h"
#include "utilities/atomic_utilities.h"

namespace Kratos
{







template<class TSparseSpace,
class TDenseSpace, 
class TLinearSolver 
>
class ResidualBasedBlockBuilderAndSolverWithLagrangeMultiplier
: public ResidualBasedBlockBuilderAndSolver< TSparseSpace, TDenseSpace, TLinearSolver >
{
public:

KRATOS_DEFINE_LOCAL_FLAG( DOUBLE_LAGRANGE_MULTIPLIER );
KRATOS_DEFINE_LOCAL_FLAG( TRANSFORMATION_MATRIX_COMPUTED );

enum class CONSTRAINT_FACTOR {CONSIDER_NORM_DIAGONAL_CONSTRAINT_FACTOR = 0, CONSIDER_MEAN_DIAGONAL_CONSTRAINT_FACTOR = 1, CONSIDER_PRESCRIBED_CONSTRAINT_FACTOR = 2};
enum class AUXILIAR_CONSTRAINT_FACTOR {CONSIDER_NORM_DIAGONAL_CONSTRAINT_FACTOR = 0, CONSIDER_MEAN_DIAGONAL_CONSTRAINT_FACTOR = 1, CONSIDER_PRESCRIBED_CONSTRAINT_FACTOR = 2};

KRATOS_CLASS_POINTER_DEFINITION(ResidualBasedBlockBuilderAndSolverWithLagrangeMultiplier);

typedef BuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver>   BaseBuilderAndSolverType;
typedef ResidualBasedBlockBuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver> BaseType;

typedef ResidualBasedBlockBuilderAndSolverWithLagrangeMultiplier<TSparseSpace, TDenseSpace, TLinearSolver> ClassType;

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

typedef Node NodeType;
typedef typename NodeType::DofType DofType;
typedef typename DofType::Pointer DofPointerType;



explicit ResidualBasedBlockBuilderAndSolverWithLagrangeMultiplier() : BaseType()
{
}


explicit ResidualBasedBlockBuilderAndSolverWithLagrangeMultiplier(
typename TLinearSolver::Pointer pNewLinearSystemSolver,
Parameters ThisParameters
) : BaseType(pNewLinearSystemSolver)
{
ThisParameters = this->ValidateAndAssignParameters(ThisParameters, this->GetDefaultParameters());
this->AssignSettings(ThisParameters);
}


explicit ResidualBasedBlockBuilderAndSolverWithLagrangeMultiplier(typename TLinearSolver::Pointer pNewLinearSystemSolver)
: BaseType(pNewLinearSystemSolver)
{
BaseType::mScalingDiagonal = SCALING_DIAGONAL::NO_SCALING;
BaseType::mOptions.Set(BaseType::SILENT_WARNINGS, false);
mConstraintFactorConsidered = CONSTRAINT_FACTOR::CONSIDER_NORM_DIAGONAL_CONSTRAINT_FACTOR;
mAuxiliarConstraintFactorConsidered = AUXILIAR_CONSTRAINT_FACTOR::CONSIDER_NORM_DIAGONAL_CONSTRAINT_FACTOR;
BaseType::mOptions.Set(DOUBLE_LAGRANGE_MULTIPLIER, true);
BaseType::mOptions.Set(TRANSFORMATION_MATRIX_COMPUTED, false);
}


~ResidualBasedBlockBuilderAndSolverWithLagrangeMultiplier() override
{
}


typename BaseBuilderAndSolverType::Pointer Create(
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

if (rA.size1() != BaseType::mEquationSystemSize || rA.size2() != BaseType::mEquationSystemSize) {
rA.resize(BaseType::mEquationSystemSize, BaseType::mEquationSystemSize, false);
BaseType::ConstructMatrixStructure(pScheme, rA, rModelPart);
}

BaseType::Build(pScheme, rModelPart, rA, rb);

KRATOS_CATCH("")
}


void SystemSolve(
TSystemMatrixType& rA,
TSystemVectorType& rDx,
TSystemVectorType& rb
) override
{
KRATOS_TRY

const double norm_b = (TSparseSpace::Size(rb) != 0) ? TSparseSpace::TwoNorm(rb) : 0.0;
if (norm_b < std::numeric_limits<double>::epsilon()) {
BaseType::mpLinearSystemSolver->Solve(rA, rDx, rb);
} else {
TSparseSpace::SetToZero(rDx);
}

KRATOS_INFO_IF("ResidualBasedBlockBuilderAndSolverWithLagrangeMultiplier", this->GetEchoLevel() > 1) << *(BaseType::mpLinearSystemSolver) << std::endl;

KRATOS_CATCH("")
}


void SystemSolveWithPhysics(
TSystemMatrixType& rA,
TSystemVectorType& rDx,
TSystemVectorType& rb,
ModelPart& rModelPart
) override
{
BaseType::InternalSystemSolveWithPhysics(rA, rDx, rb, rModelPart);
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

const SizeType total_system_size = (BaseType::mOptions.Is(DOUBLE_LAGRANGE_MULTIPLIER)) ? BaseType::mEquationSystemSize + 2 * BaseType::mSlaveIds.size() : BaseType::mEquationSystemSize + BaseType::mSlaveIds.size();
if (rDx.size() != total_system_size) {
rDx.resize(total_system_size,  false);
TSparseSpace::SetToZero(rDx);
}

BaseType::BuildAndSolve(pScheme, rModelPart, rA, rDx, rb);

IndexPartition<std::size_t>(BaseType::mSlaveIds.size()).for_each([&, this](std::size_t Index){
mLagrangeMultiplierVector[Index] += rDx[this->mEquationSystemSize + Index];
});

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

const SizeType total_system_size = (BaseType::mOptions.Is(DOUBLE_LAGRANGE_MULTIPLIER)) ? BaseType::mEquationSystemSize + 2 * BaseType::mSlaveIds.size() : BaseType::mEquationSystemSize + BaseType::mSlaveIds.size();
if (rDx.size() != total_system_size) {
rDx.resize(total_system_size,  false);
TSparseSpace::SetToZero(rDx);
}

BaseType::BuildRHSAndSolve(pScheme, rModelPart, rA, rDx, rb);

IndexPartition<std::size_t>(BaseType::mSlaveIds.size()).for_each([&, this](std::size_t Index){
mLagrangeMultiplierVector[Index] += rDx[this->mEquationSystemSize + Index];
});

KRATOS_CATCH("")
}


void BuildRHS(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemVectorType& rb
) override
{
KRATOS_TRY

if (rb.size() != BaseType::mEquationSystemSize) {
rb.resize(BaseType::mEquationSystemSize, false);
}

BaseType::BuildRHS(pScheme, rModelPart, rb);

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
TSparseSpace::SetToZero(rb);

BaseType::BuildRHSNoDirichlet(pScheme, rModelPart, rb);

const auto it_dof_begin = BaseType::mDofSet.begin();

block_for_each(BaseType::mDofSet, [&](Dof<double>& rDof){
if (rDof.IsFixed()) {
rDof.GetSolutionStepReactionValue() = -rb[rDof.EquationId()];
}
});

IndexPartition<std::size_t>(BaseType::mSlaveIds.size()).for_each([&, this](std::size_t Index){
const IndexType equation_id = this->mSlaveIds[Index];
auto it_dof = it_dof_begin + equation_id;
it_dof->GetSolutionStepReactionValue() = mLagrangeMultiplierVector[mCorrespondanceDofsSlave[equation_id]];
});
}


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
const std::size_t ndofs = BaseType::mDofSet.size();

IndexPartition<std::size_t>(ndofs).for_each([&](std::size_t Index){
auto it_dof_iterator = it_dof_iterator_begin + Index;
if (it_dof_iterator->IsFixed()) {
scaling_factors[Index] = 0.0;
} else {
scaling_factors[Index] = 1.0;
}
});

const std::size_t loop_size = system_size - ndofs;

IndexPartition<std::size_t>(loop_size).for_each([&](std::size_t Index){
scaling_factors[ndofs + Index] = 1.0;
});


double* Avalues = rA.value_data().begin();
std::size_t* Arow_indices = rA.index1_data().begin();
std::size_t* Acol_indices = rA.index2_data().begin();

const double zero_tolerance = std::numeric_limits<double>::epsilon();

BaseType::mScaleFactor = TSparseSpace::GetScaleNorm(rModelPart.GetProcessInfo(), rA, BaseType::mScalingDiagonal);

IndexPartition<std::size_t>(system_size).for_each([&](std::size_t Index){
std::size_t col_begin = 0, col_end  = 0;
bool empty = true;

col_begin = Arow_indices[Index];
col_end = Arow_indices[Index + 1];
empty = true;
for (std::size_t j = col_begin; j < col_end; ++j) {
if(std::abs(Avalues[j]) > zero_tolerance) {
empty = false;
break;
}
}

if(empty) {
rA(Index, Index) = this->mScaleFactor;
rb[Index] = 0.0;
}
});

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
if (mConstraintFactorConsidered != CONSTRAINT_FACTOR::CONSIDER_PRESCRIBED_CONSTRAINT_FACTOR) {
TSystemMatrixType A(BaseType::mEquationSystemSize, BaseType::mEquationSystemSize);
BaseType::ConstructMatrixStructure(pScheme, A, rModelPart);
this->BuildLHS(pScheme, rModelPart, A);
const double constraint_scale_factor = mConstraintFactorConsidered == CONSTRAINT_FACTOR::CONSIDER_NORM_DIAGONAL_CONSTRAINT_FACTOR ? TSparseSpace::GetMaxDiagonal(A) : TSparseSpace::GetDiagonalNorm(A);
mConstraintFactor = constraint_scale_factor;
}

if (TSparseSpace::Size1(BaseType::mT) != BaseType::mSlaveIds.size() || TSparseSpace::Size2(BaseType::mT) != BaseType::mEquationSystemSize) {
BaseType::mT.resize(BaseType::mSlaveIds.size(), BaseType::mEquationSystemSize, false);
ConstructMasterSlaveConstraintsStructure(rModelPart);
}

if (BaseType::mOptions.IsNot(TRANSFORMATION_MATRIX_COMPUTED)) {
BuildMasterSlaveConstraints(rModelPart);
}

const SizeType number_of_slave_dofs = TSparseSpace::Size1(BaseType::mT);

const SizeType total_size_of_system = BaseType::mEquationSystemSize + (BaseType::mOptions.Is(DOUBLE_LAGRANGE_MULTIPLIER) ? 2 * number_of_slave_dofs : number_of_slave_dofs);
TSystemVectorType b_modified(total_size_of_system);

IndexPartition<std::size_t>(this->mEquationSystemSize).for_each([&](std::size_t Index){
b_modified[Index] = rb[Index];
});

const SizeType loop_size = total_size_of_system - BaseType::mEquationSystemSize;
const SizeType start_index = BaseType::mEquationSystemSize;

IndexPartition<std::size_t>(loop_size).for_each([&](std::size_t Index){
b_modified[start_index + Index] = 0.0;
});

rb.resize(total_size_of_system, false);

TSystemVectorType b_lm(total_size_of_system);
ComputeRHSLMContributions(b_lm, mConstraintFactor);

TSparseSpace::UnaliasedAdd(b_modified, 1.0, b_lm);

TSparseSpace::Copy(b_modified, rb);
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
const ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();

BuildMasterSlaveConstraints(rModelPart);

TSystemMatrixType copy_of_A;
copy_of_A.swap(rA);

TSystemMatrixType copy_of_T(BaseType::mT);
TSystemMatrixType transpose_of_T(TSparseSpace::Size2(BaseType::mT), TSparseSpace::Size1(BaseType::mT));
SparseMatrixMultiplicationUtility::TransposeMatrix<TSystemMatrixType, TSystemMatrixType>(transpose_of_T, BaseType::mT);

const SizeType number_of_slave_dofs = TSparseSpace::Size1(BaseType::mT);

const SizeType total_size_of_system = BaseType::mEquationSystemSize + (BaseType::mOptions.Is(DOUBLE_LAGRANGE_MULTIPLIER) ? 2 * number_of_slave_dofs : number_of_slave_dofs);
TSystemVectorType b_modified(total_size_of_system);

IndexPartition<std::size_t>(this->mEquationSystemSize).for_each([&](std::size_t Index){
b_modified[Index] = rb[Index];
});

auto loop_size = static_cast<int>(total_size_of_system) - static_cast<int>(BaseType::mEquationSystemSize);
auto start_index = BaseType::mEquationSystemSize;

IndexPartition<std::size_t>(loop_size).for_each([&](std::size_t Index){
b_modified[start_index + Index] = 0.0;
});
rb.resize(total_size_of_system, false);

const SizeType number_of_blocks = BaseType::mOptions.Is(DOUBLE_LAGRANGE_MULTIPLIER) ? 3 : 2;

DenseMatrix<TSystemMatrixType*> matrices_p_blocks(number_of_blocks, number_of_blocks);
DenseMatrix<double> contribution_coefficients(number_of_blocks, number_of_blocks);
DenseMatrix<bool> transpose_blocks(number_of_blocks, number_of_blocks);

const bool has_constraint_scale_factor = mConstraintFactorConsidered == CONSTRAINT_FACTOR::CONSIDER_PRESCRIBED_CONSTRAINT_FACTOR ? true : false;
KRATOS_ERROR_IF(has_constraint_scale_factor && !r_current_process_info.Has(CONSTRAINT_SCALE_FACTOR)) << "Constraint scale factor not defined at process info" << std::endl;
const double constraint_scale_factor = has_constraint_scale_factor ? r_current_process_info.GetValue(CONSTRAINT_SCALE_FACTOR) : mConstraintFactorConsidered == CONSTRAINT_FACTOR::CONSIDER_NORM_DIAGONAL_CONSTRAINT_FACTOR ? TSparseSpace::GetDiagonalNorm(copy_of_A) : TSparseSpace::GetAveragevalueDiagonal(copy_of_A);
mConstraintFactor = constraint_scale_factor;


matrices_p_blocks(0,0) = &copy_of_A;
matrices_p_blocks(0,1) = &transpose_of_T;
matrices_p_blocks(1,0) = &copy_of_T;

contribution_coefficients(0, 0) = 1.0;
contribution_coefficients(0, 1) = mConstraintFactor;
contribution_coefficients(1, 0) = mConstraintFactor;

for (IndexType i = 0; i < number_of_blocks; ++i) {
for (IndexType j = 0; j < number_of_blocks; ++j) {
transpose_blocks(i, j) = false;
}
}

if (BaseType::mOptions.Is(DOUBLE_LAGRANGE_MULTIPLIER)) {
const bool has_auxiliar_constraint_scale_factor = mAuxiliarConstraintFactorConsidered == AUXILIAR_CONSTRAINT_FACTOR::CONSIDER_PRESCRIBED_CONSTRAINT_FACTOR ? true : false;
KRATOS_ERROR_IF(has_auxiliar_constraint_scale_factor && !r_current_process_info.Has(AUXILIAR_CONSTRAINT_SCALE_FACTOR)) << "Auxiliar constraint scale factor not defined at process info" << std::endl;
const double auxiliar_constraint_scale_factor = has_auxiliar_constraint_scale_factor ? r_current_process_info.GetValue(AUXILIAR_CONSTRAINT_SCALE_FACTOR) : mAuxiliarConstraintFactorConsidered == AUXILIAR_CONSTRAINT_FACTOR::CONSIDER_NORM_DIAGONAL_CONSTRAINT_FACTOR ? TSparseSpace::GetDiagonalNorm(copy_of_A) : TSparseSpace::GetAveragevalueDiagonal(copy_of_A);
mAuxiliarConstraintFactor = auxiliar_constraint_scale_factor;

TSystemMatrixType identity_matrix(number_of_slave_dofs, number_of_slave_dofs);
for (IndexType i = 0; i < number_of_slave_dofs; ++i) {
identity_matrix.push_back(i, i, 1.0);
}

KRATOS_ERROR_IF_NOT(identity_matrix.nnz() == number_of_slave_dofs) << "Inconsistent number of non-zero values in the identity matrix: " << number_of_slave_dofs << " vs " << identity_matrix.nnz() << std::endl;

matrices_p_blocks(0,2) = &transpose_of_T;
matrices_p_blocks(2,0) = &copy_of_T;
matrices_p_blocks(1,1) = &identity_matrix;
matrices_p_blocks(1,2) = &identity_matrix;
matrices_p_blocks(2,1) = &identity_matrix;
matrices_p_blocks(2,2) = &identity_matrix;

contribution_coefficients(0, 2) = mConstraintFactor;
contribution_coefficients(2, 0) = mConstraintFactor;
contribution_coefficients(1, 1) = -mAuxiliarConstraintFactor;
contribution_coefficients(1, 2) = mAuxiliarConstraintFactor;
contribution_coefficients(2, 1) = mAuxiliarConstraintFactor;
contribution_coefficients(2, 2) = -mAuxiliarConstraintFactor;

SparseMatrixMultiplicationUtility::AssembleSparseMatrixByBlocks(rA, matrices_p_blocks, contribution_coefficients, transpose_blocks);
} else {
TSystemMatrixType zero_matrix(number_of_slave_dofs, number_of_slave_dofs);

matrices_p_blocks(1,1) = &zero_matrix;

contribution_coefficients(1, 1) = 0.0;

SparseMatrixMultiplicationUtility::AssembleSparseMatrixByBlocks(rA, matrices_p_blocks, contribution_coefficients, transpose_blocks);
}

TSystemVectorType b_lm(total_size_of_system);
ComputeRHSLMContributions(b_lm, constraint_scale_factor);

TSparseSpace::UnaliasedAdd(b_modified, 1.0, b_lm);

TSparseSpace::Copy(b_modified, rb);
}

KRATOS_CATCH("")
}


void Clear() override
{
BaseType::Clear();

mCorrespondanceDofsSlave.clear();
mLagrangeMultiplierVector.resize(0,false);

BaseType::mOptions.Set(TRANSFORMATION_MATRIX_COMPUTED, false);
}


int Check(ModelPart& rModelPart) override
{
KRATOS_TRY

return BaseType::Check(rModelPart);

KRATOS_CATCH("");
}


Parameters GetDefaultParameters() const override
{
Parameters default_parameters = Parameters(R"(
{
"name"                                               : "ResidualBasedBlockBuilderAndSolverWithLagrangeMultiplier",
"consider_lagrange_multiplier_constraint_resolution" : "double",
"constraint_scale_factor"                            : "use_mean_diagonal",
"auxiliar_constraint_scale_factor"                   : "use_mean_diagonal"
})");

const Parameters base_default_parameters = BaseType::GetDefaultParameters();
default_parameters.RecursivelyAddMissingParameters(base_default_parameters);
return default_parameters;
}


static std::string Name()
{
return "block_builder_and_solver_with_lagrange_multiplier";
}




std::string Info() const override
{
return "ResidualBasedBlockBuilderAndSolverWithLagrangeMultiplier";
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


std::unordered_map<IndexType, IndexType> mCorrespondanceDofsSlave; 
TSystemVectorType mLagrangeMultiplierVector;                       
double mConstraintFactor = 0.0;                                    
double mAuxiliarConstraintFactor = 0.0;                            

CONSTRAINT_FACTOR mConstraintFactorConsidered;                  
AUXILIAR_CONSTRAINT_FACTOR mAuxiliarConstraintFactorConsidered; 




void ConstructMasterSlaveConstraintsStructure(ModelPart& rModelPart) override
{
KRATOS_TRY

if (rModelPart.MasterSlaveConstraints().size() > 0) {
Timer::Start("ConstraintsRelationMatrixStructure");
const ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();

DofsVectorType slave_dof_list, master_dof_list;

const auto it_const_begin = rModelPart.MasterSlaveConstraints().begin();

const std::size_t size_indices = BaseType::mDofSet.size();
std::vector<std::unordered_set<IndexType>> indices(size_indices);

std::vector<LockObject> lock_array(size_indices);

#pragma omp parallel firstprivate(slave_dof_list, master_dof_list)
{
Element::EquationIdVectorType slave_ids(3);
Element::EquationIdVectorType master_ids(3);
std::unordered_map<IndexType, std::unordered_set<IndexType>> temp_indices;

#pragma omp for schedule(guided, 512) nowait
for (int i_const = 0; i_const < static_cast<int>(rModelPart.MasterSlaveConstraints().size()); ++i_const) {
auto it_const = it_const_begin + i_const;

if(it_const->IsActive()) {
it_const->EquationIdVector(slave_ids, master_ids, r_current_process_info);

for (auto &id_i : slave_ids) {
temp_indices[id_i].insert(id_i);
temp_indices[id_i].insert(master_ids.begin(), master_ids.end());
}
}
}

for (int i = 0; i < static_cast<int>(temp_indices.size()); ++i) {
lock_array[i].lock();
indices[i].insert(temp_indices[i].begin(), temp_indices[i].end());
lock_array[i].unlock();
}
}

IndexType counter = 0;
mCorrespondanceDofsSlave.clear();
BaseType::mSlaveIds.clear();
BaseType::mMasterIds.clear();
for (int i = 0; i < static_cast<int>(size_indices); ++i) {
if (indices[i].size() == 0) { 
BaseType::mMasterIds.push_back(i);
} else { 
BaseType::mSlaveIds.push_back(i);
mCorrespondanceDofsSlave.insert(std::pair<IndexType, IndexType>(i, counter));
++counter;
}
}

const std::size_t slave_size = BaseType::mSlaveIds.size();

std::size_t nnz = 0;
nnz = IndexPartition<std::size_t>(slave_size).for_each<SumReduction<std::size_t>>([&, this](std::size_t Index){
return indices[this->mSlaveIds[Index]].size();
});

BaseType::mT = TSystemMatrixType(slave_size, size_indices, nnz);
BaseType::mConstantVector.resize(slave_size, false);
mLagrangeMultiplierVector.resize(slave_size, false);
TSparseSpace::SetToZero(mLagrangeMultiplierVector);

double *Tvalues = BaseType::mT.value_data().begin();
IndexType *Trow_indices = BaseType::mT.index1_data().begin();
IndexType *Tcol_indices = BaseType::mT.index2_data().begin();

Trow_indices[0] = 0;
for (int i = 0; i < static_cast<int>(slave_size); ++i) {
Trow_indices[i + 1] = Trow_indices[i] + indices[BaseType::mSlaveIds[i]].size();
}

IndexPartition<std::size_t>(slave_size).for_each([&, this](std::size_t Index){
const IndexType row_begin = Trow_indices[Index];
const IndexType row_end = Trow_indices[Index + 1];
IndexType k = row_begin;
const IndexType i_slave = this->mSlaveIds[Index];
for (auto it = indices[i_slave].begin(); it != indices[i_slave].end(); ++it) {
Tcol_indices[k] = *it;
Tvalues[k] = 0.0;
++k;
}

indices[i_slave].clear(); 

std::sort(&Tcol_indices[row_begin], &Tcol_indices[row_end]);
});

BaseType::mT.set_filled(slave_size + 1, nnz);

BaseType::mOptions.Set(TRANSFORMATION_MATRIX_COMPUTED, false);

Timer::Stop("ConstraintsRelationMatrixStructure");
}

KRATOS_CATCH("")
}


void BuildMasterSlaveConstraints(ModelPart& rModelPart) override
{
KRATOS_TRY

TSparseSpace::SetToZero(BaseType::mT);
TSparseSpace::SetToZero(BaseType::mConstantVector);

const ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();

DofsVectorType slave_dof_list, master_dof_list;

Matrix transformation_matrix = LocalSystemMatrixType(0, 0);
Vector constant_vector = LocalSystemVectorType(0);

Element::EquationIdVectorType slave_equation_ids, master_equation_ids;

const int number_of_constraints = static_cast<int>(rModelPart.MasterSlaveConstraints().size());

#pragma omp parallel firstprivate(transformation_matrix, constant_vector, slave_equation_ids, master_equation_ids)
{
#pragma omp for schedule(guided, 512)
for (int i_const = 0; i_const < number_of_constraints; ++i_const) {
auto it_const = rModelPart.MasterSlaveConstraints().begin() + i_const;

if (it_const->IsActive()) {
it_const->CalculateLocalSystem(transformation_matrix, constant_vector, r_current_process_info);
it_const->EquationIdVector(slave_equation_ids, master_equation_ids, r_current_process_info);

for (IndexType i = 0; i < slave_equation_ids.size(); ++i) {
const IndexType i_global = mCorrespondanceDofsSlave[slave_equation_ids[i]];

BaseType::AssembleRowContribution(BaseType::mT, - transformation_matrix, i_global, i, master_equation_ids);

const double constant_value = constant_vector[i];
double& r_value = BaseType::mConstantVector[i_global];
AtomicAdd(r_value, constant_value);
}
}
}
}

for (auto eq_id : BaseType::mSlaveIds) {
BaseType::mT(mCorrespondanceDofsSlave[eq_id], eq_id) = 1.0;
}

BaseType::mOptions.Set(TRANSFORMATION_MATRIX_COMPUTED, true);

KRATOS_CATCH("")
}


Parameters ValidateAndAssignParameters(
Parameters ThisParameters,
const Parameters DefaultParameters
) const override
{
ThisParameters.RecursivelyValidateAndAssignDefaults(DefaultParameters);
return ThisParameters;
}


void AssignSettings(const Parameters ThisParameters) override
{
BaseType::AssignSettings(ThisParameters);

std::set<std::string> available_options_for_constraints_scale = {"use_mean_diagonal","use_diagonal_norm","defined_in_process_info"};

const std::string& r_constraint_scale_factor = ThisParameters["constraint_scale_factor"].GetString();

if (available_options_for_constraints_scale.find(r_constraint_scale_factor) == available_options_for_constraints_scale.end()) {
std::stringstream msg;
msg << "Currently prescribed constraint scale factor : " << r_constraint_scale_factor << "\n";
msg << "Admissible values for the constraint scale factor are : use_mean_diagonal, use_diagonal_norm, or defined_in_process_info" << "\n";
KRATOS_ERROR << msg.str() << std::endl;
}

if (r_constraint_scale_factor == "use_mean_diagonal") {
mConstraintFactorConsidered = CONSTRAINT_FACTOR::CONSIDER_MEAN_DIAGONAL_CONSTRAINT_FACTOR;
} else if (r_constraint_scale_factor == "use_diagonal_norm") { 
mConstraintFactorConsidered = CONSTRAINT_FACTOR::CONSIDER_NORM_DIAGONAL_CONSTRAINT_FACTOR;
} else { 
mConstraintFactorConsidered = CONSTRAINT_FACTOR::CONSIDER_PRESCRIBED_CONSTRAINT_FACTOR;
}

const std::string& r_auxiliar_constraint_scale_factor = ThisParameters["auxiliar_constraint_scale_factor"].GetString();

if (available_options_for_constraints_scale.find(r_auxiliar_constraint_scale_factor) == available_options_for_constraints_scale.end()) {
std::stringstream msg;
msg << "Currently prescribed constraint scale factor : " << r_auxiliar_constraint_scale_factor << "\n";
msg << "Admissible values for the constraint scale factor are : use_mean_diagonal, use_diagonal_norm, or defined_in_process_info" << "\n";
KRATOS_ERROR << msg.str() << std::endl;
}

if (r_auxiliar_constraint_scale_factor == "use_mean_diagonal") {
mAuxiliarConstraintFactorConsidered = AUXILIAR_CONSTRAINT_FACTOR::CONSIDER_MEAN_DIAGONAL_CONSTRAINT_FACTOR;
} else if (r_auxiliar_constraint_scale_factor == "use_diagonal_norm") { 
mAuxiliarConstraintFactorConsidered = AUXILIAR_CONSTRAINT_FACTOR::CONSIDER_NORM_DIAGONAL_CONSTRAINT_FACTOR;
} else { 
mAuxiliarConstraintFactorConsidered = AUXILIAR_CONSTRAINT_FACTOR::CONSIDER_PRESCRIBED_CONSTRAINT_FACTOR;
}

if (ThisParameters["consider_lagrange_multiplier_constraint_resolution"].GetString() == "double") {
BaseType::mOptions.Set(DOUBLE_LAGRANGE_MULTIPLIER, true);
} else {
BaseType::mOptions.Set(DOUBLE_LAGRANGE_MULTIPLIER, false);
}

BaseType::mOptions.Set(TRANSFORMATION_MATRIX_COMPUTED, false);
}





private:





void ComputeRHSLMContributions(
TSystemVectorType& rbLM,
const double ScaleFactor = 1.0
)
{
KRATOS_TRY

const auto it_dof_begin = BaseType::mDofSet.begin();
const int ndofs = static_cast<int>(BaseType::mDofSet.size());

const SizeType number_of_slave_dofs = TSparseSpace::Size1(BaseType::mT);
const SizeType total_size_of_system = BaseType::mEquationSystemSize + (BaseType::mOptions.Is(DOUBLE_LAGRANGE_MULTIPLIER) ? 2 * number_of_slave_dofs : number_of_slave_dofs);
if (TSparseSpace::Size(rbLM) != total_size_of_system)
rbLM.resize(total_size_of_system, false);
TSystemVectorType aux_lm_rhs_contribution(number_of_slave_dofs);
TSystemVectorType aux_whole_dof_vector(ndofs);

IndexPartition<std::size_t>(ndofs).for_each([&](std::size_t Index){
auto it_dof = it_dof_begin + Index;
aux_whole_dof_vector[Index] = it_dof->GetSolutionStepValue();
});

TSystemVectorType aux_slave_dof_vector(number_of_slave_dofs);
TSparseSpace::Mult(BaseType::mT, aux_whole_dof_vector, aux_slave_dof_vector);

noalias(aux_lm_rhs_contribution) = ScaleFactor * (BaseType::mConstantVector -  aux_slave_dof_vector);

if (BaseType::mOptions.Is(DOUBLE_LAGRANGE_MULTIPLIER)) {
IndexPartition<std::size_t>(number_of_slave_dofs).for_each([&](std::size_t Index){
rbLM[ndofs + Index] = aux_lm_rhs_contribution[Index];
rbLM[ndofs + number_of_slave_dofs + Index] = aux_lm_rhs_contribution[Index];
});
} else {
IndexPartition<std::size_t>(number_of_slave_dofs).for_each([&](std::size_t Index){
rbLM[ndofs + Index] = aux_lm_rhs_contribution[Index];
});
}

TSystemMatrixType T_transpose_matrix(ndofs, number_of_slave_dofs);
SparseMatrixMultiplicationUtility::TransposeMatrix<TSystemMatrixType, TSystemMatrixType>(T_transpose_matrix, BaseType::mT, -ScaleFactor);

TSparseSpace::Mult(T_transpose_matrix, mLagrangeMultiplierVector, aux_whole_dof_vector);
IndexPartition<std::size_t>(ndofs).for_each([&](std::size_t Index){
rbLM[Index] = aux_whole_dof_vector[Index];
});

KRATOS_CATCH("")
}






}; 



template<class TSparseSpace, class TDenseSpace, class TLinearSolver>
const Kratos::Flags ResidualBasedBlockBuilderAndSolverWithLagrangeMultiplier<TSparseSpace, TDenseSpace, TLinearSolver>::DOUBLE_LAGRANGE_MULTIPLIER(Kratos::Flags::Create(1));
template<class TSparseSpace, class TDenseSpace, class TLinearSolver>
const Kratos::Flags ResidualBasedBlockBuilderAndSolverWithLagrangeMultiplier<TSparseSpace, TDenseSpace, TLinearSolver>::TRANSFORMATION_MATRIX_COMPUTED(Kratos::Flags::Create(2));


} 

#endif 
