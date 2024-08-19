
#pragma once

#include <string>
#include <iostream>
#include <sstream>
#include <cstddef>


#include "includes/define.h"
#include "includes/model_part.h"
#include "linear_solvers/reorderer.h"
#include "linear_solvers/iterative_solver.h"
#include "utilities/parallel_utilities.h"
#include "contact_structural_mechanics_application_variables.h"
#include "utilities/sparse_matrix_multiplication_utility.h"
#include "custom_utilities/logging_settings.hpp"

namespace Kratos
{


template<class TSparseSpaceType, class TDenseSpaceType,
class TPreconditionerType = Preconditioner<TSparseSpaceType, TDenseSpaceType>,
class TReordererType = Reorderer<TSparseSpaceType, TDenseSpaceType> >
class MixedULMLinearSolver :
public IterativeSolver<TSparseSpaceType, TDenseSpaceType,TPreconditionerType, TReordererType>
{
public:

enum class BlockType {
OTHER,
MASTER,
SLAVE_INACTIVE,
SLAVE_ACTIVE,
LM_INACTIVE,
LM_ACTIVE
};


KRATOS_DEFINE_LOCAL_FLAG( BLOCKS_ARE_ALLOCATED );

KRATOS_DEFINE_LOCAL_FLAG( IS_INITIALIZED );

KRATOS_CLASS_POINTER_DEFINITION (MixedULMLinearSolver);

typedef IterativeSolver<TSparseSpaceType, TDenseSpaceType, TPreconditionerType, TReordererType> BaseType;

typedef LinearSolver<TSparseSpaceType, TDenseSpaceType, TReordererType> LinearSolverType;

typedef typename LinearSolverType::Pointer LinearSolverPointerType;

typedef typename TSparseSpaceType::MatrixType SparseMatrixType;

typedef typename TSparseSpaceType::VectorType VectorType;

typedef typename TDenseSpaceType::MatrixType DenseMatrixType;

typedef typename TDenseSpaceType::VectorType DenseVectorType;

typedef Node NodeType;

typedef typename ModelPart::DofType DofType;

typedef typename ModelPart::DofsArrayType DofsArrayType;

typedef ModelPart::ConditionsContainerType ConditionsArrayType;

typedef ModelPart::NodesContainerType NodesArrayType;

typedef std::size_t SizeType;

typedef std::size_t IndexType;

typedef DenseVector<IndexType> IndexVectorType;

typedef DenseVector<BlockType> BlockTypeVectorType;

static constexpr double ZeroTolerance = std::numeric_limits<double>::epsilon();



MixedULMLinearSolver (
LinearSolverPointerType pSolverDispBlock,
const double MaxTolerance,
const std::size_t MaxIterationNumber
) : BaseType (MaxTolerance, MaxIterationNumber),
mpSolverDispBlock(pSolverDispBlock)
{
mOptions.Set(BLOCKS_ARE_ALLOCATED, false);
mOptions.Set(IS_INITIALIZED, false);
}



MixedULMLinearSolver(
LinearSolverPointerType pSolverDispBlock,
Parameters ThisParameters =  Parameters(R"({})")
): BaseType (),
mpSolverDispBlock(pSolverDispBlock)

{
KRATOS_TRY

Parameters default_parameters = GetDefaultParameters();
ThisParameters.ValidateAndAssignDefaults(default_parameters);

this->SetTolerance( ThisParameters["tolerance"].GetDouble() );
this->SetMaxIterationsNumber( ThisParameters["max_iteration_number"].GetInt() );
mEchoLevel = ThisParameters["echo_level"].GetInt();
mOptions.Set(BLOCKS_ARE_ALLOCATED, false);
mOptions.Set(IS_INITIALIZED, false);

KRATOS_CATCH("")
}


MixedULMLinearSolver (const MixedULMLinearSolver& rOther)
: BaseType(rOther),
mpSolverDispBlock(rOther.mpSolverDispBlock),
mOptions(rOther.mOptions),
mDisplacementDofs(rOther.mDisplacementDofs),
mMasterIndices(rOther.mMasterIndices),
mSlaveInactiveIndices(rOther.mSlaveInactiveIndices),
mSlaveActiveIndices(rOther.mSlaveActiveIndices),
mLMInactiveIndices(rOther.mLMInactiveIndices),
mLMActiveIndices(rOther.mLMActiveIndices),
mOtherIndices(rOther.mOtherIndices),
mGlobalToLocalIndexing(rOther.mGlobalToLocalIndexing),
mWhichBlockType(rOther.mWhichBlockType),
mKDispModified(rOther.mKDispModified),
mKLMAModified(rOther.mKLMAModified),
mKLMIModified(rOther.mKLMIModified),
mKSAN(rOther.mKSAN),
mKSAM(rOther.mKSAM),
mKSASI(rOther.mKSASI),
mKSASA(rOther.mKSASA),
mPOperator(rOther.mPOperator),
mCOperator(rOther.mCOperator),
mResidualLMActive(rOther.mResidualLMActive),
mResidualLMInactive(rOther.mResidualLMInactive),
mResidualDisp(rOther.mResidualDisp),
mLMActive(rOther.mLMActive),
mLMInactive(rOther.mLMInactive),
mDisp(rOther.mDisp),
mEchoLevel(rOther.mEchoLevel),
mFileCreated(rOther.mFileCreated)
{
}

~MixedULMLinearSolver() override {}


MixedULMLinearSolver& operator= (const MixedULMLinearSolver& Other)
{
return *this;
}



void Initialize (
SparseMatrixType& rA,
VectorType& rX,
VectorType& rB
) override
{
if (mOptions.Is(BLOCKS_ARE_ALLOCATED)) {
mpSolverDispBlock->Initialize(mKDispModified, mDisp, mResidualDisp);
mOptions.Set(IS_INITIALIZED, true);
} else
KRATOS_DETAIL("MixedULM Initialize") << "Linear solver intialization is deferred to the moment at which blocks are available" << std::endl;
}


void InitializeSolutionStep (
SparseMatrixType& rA,
VectorType& rX,
VectorType& rB
) override
{
if (mOptions.IsNot(BLOCKS_ARE_ALLOCATED)) {
FillBlockMatrices (true, rA, rX, rB);
mOptions.Set(BLOCKS_ARE_ALLOCATED, true);
} else {
FillBlockMatrices (false, rA, rX, rB);
mOptions.Set(BLOCKS_ARE_ALLOCATED, true);
}

if(mOptions.IsNot(IS_INITIALIZED))
this->Initialize(rA,rX,rB);

mpSolverDispBlock->InitializeSolutionStep(mKDispModified, mDisp, mResidualDisp);
}


void PerformSolutionStep (
SparseMatrixType& rA,
VectorType& rX,
VectorType& rB
) override
{
const SizeType lm_active_size = mLMActiveIndices.size();
const SizeType lm_inactive_size = mLMInactiveIndices.size();
const SizeType total_disp_size = mOtherIndices.size() + mMasterIndices.size() + mSlaveInactiveIndices.size() + mSlaveActiveIndices.size();

GetUPart (rB, mResidualDisp);

if (mDisp.size() != total_disp_size)
mDisp.resize(total_disp_size, false);
mpSolverDispBlock->Solve (mKDispModified, mDisp, mResidualDisp);

SetUPart(rX, mDisp);

if (lm_active_size > 0) {
GetLMAPart (rB, mResidualLMActive);

if (mLMActive.size() != lm_active_size)
mLMActive.resize(lm_active_size, false);
TSparseSpaceType::Mult (mKLMAModified, mResidualLMActive, mLMActive);

SetLMAPart(rX, mLMActive);
}

if (lm_inactive_size > 0) {
GetLMIPart (rB, mResidualLMInactive);

if (mLMInactive.size() != lm_inactive_size)
mLMInactive.resize(lm_inactive_size, false);
TSparseSpaceType::Mult (mKLMIModified, mResidualLMInactive, mLMInactive);

SetLMIPart(rX, mLMInactive);
}
}


void FinalizeSolutionStep (
SparseMatrixType& rA,
VectorType& rX,
VectorType& rB
) override
{
mpSolverDispBlock->FinalizeSolutionStep(mKDispModified, mDisp, mResidualDisp);
}


void Clear() override
{
mOptions.Set(BLOCKS_ARE_ALLOCATED, false);
mpSolverDispBlock->Clear();

auto& r_data_dofs = mDisplacementDofs.GetContainer(); 
for (IndexType i=0; i<r_data_dofs.size(); ++i) {
delete r_data_dofs[i];
}
r_data_dofs.clear();

mKDispModified.clear(); 
mKLMAModified.clear();  
mKLMIModified.clear();  

mKSAN.clear();  
mKSAM.clear();  
mKSASI.clear(); 
mKSASA.clear(); 

mPOperator.clear(); 
mCOperator.clear(); 

mResidualLMActive.clear();   
mResidualLMInactive.clear(); 
mResidualDisp.clear();       

mLMActive.clear();   
mLMInactive.clear(); 
mDisp.clear();       

mOptions.Set(IS_INITIALIZED, false);
}


bool Solve(
SparseMatrixType& rA,
VectorType& rX,
VectorType& rB
) override
{
if (mEchoLevel == 2) { 
KRATOS_INFO("RHS BEFORE CONDENSATION") << "RHS  = " << rB << std::endl;
} else if (mEchoLevel == 3) { 
KRATOS_INFO("LHS BEFORE CONDENSATION") << "SystemMatrix = " << rA << std::endl;
KRATOS_INFO("RHS BEFORE CONDENSATION") << "RHS  = " << rB << std::endl;
} else if (mEchoLevel >= 4) { 
const std::string matrix_market_name = "before_condensation_A_" + std::to_string(mFileCreated) + ".mm";
TSparseSpaceType::WriteMatrixMarketMatrix(matrix_market_name.c_str(), rA, false);

const std::string matrix_market_vectname = "before_condensation_b_" + std::to_string(mFileCreated) + ".mm.rhs";
TSparseSpaceType::WriteMatrixMarketVector(matrix_market_vectname.c_str(), rB);
}

if (mOptions.IsNot(IS_INITIALIZED))
this->Initialize (rA,rX,rB);

this->InitializeSolutionStep (rA,rX,rB);

this->PerformSolutionStep (rA,rX,rB);

this->FinalizeSolutionStep (rA,rX,rB);

if (mEchoLevel == 2) { 
KRATOS_INFO("Dx")  << "Solution obtained = " << mDisp << std::endl;
KRATOS_INFO("RHS") << "RHS  = " << mResidualDisp << std::endl;
} else if (mEchoLevel == 3) { 
KRATOS_INFO("LHS") << "SystemMatrix = " << mKDispModified << std::endl;
KRATOS_INFO("Dx")  << "Solution obtained = " << mDisp << std::endl;
KRATOS_INFO("RHS") << "RHS  = " << mResidualDisp << std::endl;
} else if (mEchoLevel >= 4) { 
const std::string matrix_market_name = "A_" + std::to_string(mFileCreated) + ".mm";
TSparseSpaceType::WriteMatrixMarketMatrix(matrix_market_name.c_str(), mKDispModified, false);

const std::string matrix_market_vectname = "b_" + std::to_string(mFileCreated) + ".mm.rhs";
TSparseSpaceType::WriteMatrixMarketVector(matrix_market_vectname.c_str(), mResidualDisp);
mFileCreated++;
}

return false;
}


bool Solve (
SparseMatrixType& rA,
DenseMatrixType& rX,
DenseMatrixType& rB
) override
{
return false;
}


bool AdditionalPhysicalDataIsNeeded() override
{
return true;
}


void ProvideAdditionalData (
SparseMatrixType& rA,
VectorType& rX,
VectorType& rB,
DofsArrayType& rDofSet,
ModelPart& rModelPart
) override
{
IndexType node_id;

SizeType n_lm_inactive_dofs = 0, n_lm_active_dofs = 0;
SizeType n_master_dofs = 0;
SizeType n_slave_inactive_dofs = 0, n_slave_active_dofs = 0;
SizeType tot_active_dofs = 0;

if (rModelPart.IsNot(TO_SPLIT)) {
for (auto& i_dof : rDofSet) {
node_id = i_dof.Id();
const NodeType& node = rModelPart.GetNode(node_id);
if (i_dof.EquationId() < rA.size1()) {
tot_active_dofs++;
if (IsLMDof(i_dof)) {
if (node.Is(ACTIVE))
++n_lm_active_dofs;
else
++n_lm_inactive_dofs;
} else if (node.Is(INTERFACE) && IsDisplacementDof(i_dof)) {
if (node.Is(MASTER)) {
++n_master_dofs;
} else if (node.Is(SLAVE)) {
if (node.Is(ACTIVE))
++n_slave_active_dofs;
else
++n_slave_inactive_dofs;
}
}
}
}
} else {
for (auto& i_dof : rDofSet) {
node_id = i_dof.Id();
const NodeType& node = rModelPart.GetNode(node_id);
tot_active_dofs++;
if (IsLMDof(i_dof)) {
if (node.Is(ACTIVE))
++n_lm_active_dofs;
else
++n_lm_inactive_dofs;
} else if (node.Is(INTERFACE) && IsDisplacementDof(i_dof)) {
if (node.Is(MASTER)) {
++n_master_dofs;
} else if (node.Is(SLAVE)) {
if (node.Is(ACTIVE))
++n_slave_active_dofs;
else
++n_slave_inactive_dofs;
}
}
}
}

KRATOS_ERROR_IF(tot_active_dofs != rA.size1()) << "Total system size does not coincide with the free dof map: " << tot_active_dofs << " vs " << rA.size1() << std::endl;

if (mMasterIndices.size() != n_master_dofs)
mMasterIndices.resize (n_master_dofs,false);
if (mSlaveInactiveIndices.size() != n_slave_inactive_dofs)
mSlaveInactiveIndices.resize (n_slave_inactive_dofs,false);
if (mSlaveActiveIndices.size() != n_slave_active_dofs)
mSlaveActiveIndices.resize (n_slave_active_dofs,false);
if (mLMInactiveIndices.size() != n_lm_inactive_dofs)
mLMInactiveIndices.resize (n_lm_inactive_dofs,false);
if (mLMActiveIndices.size() != n_lm_active_dofs)
mLMActiveIndices.resize (n_lm_active_dofs,false);

const SizeType n_other_dofs = tot_active_dofs - n_lm_inactive_dofs - n_lm_active_dofs - n_master_dofs - n_slave_inactive_dofs - n_slave_active_dofs;
if (mOtherIndices.size() != n_other_dofs)
mOtherIndices.resize (n_other_dofs, false);
if (mGlobalToLocalIndexing.size() != tot_active_dofs)
mGlobalToLocalIndexing.resize (tot_active_dofs,false);
if (mWhichBlockType.size() != tot_active_dofs)
mWhichBlockType.resize(tot_active_dofs, false);

KRATOS_ERROR_IF_NOT(n_lm_active_dofs == n_slave_active_dofs) << "The number of active LM dofs: " << n_lm_active_dofs << " and active slave nodes dofs: " << n_slave_active_dofs << " does not coincide" << std::endl;


SizeType lm_inactive_counter = 0, lm_active_counter = 0;
SizeType master_counter = 0;
SizeType slave_inactive_counter = 0, slave_active_counter = 0;
SizeType other_counter = 0;
IndexType global_pos = 0;

if (rModelPart.IsNot(TO_SPLIT)) {
for (auto& i_dof : rDofSet) {
node_id = i_dof.Id();
const NodeType& r_node = rModelPart.GetNode(node_id);
if (i_dof.EquationId() < rA.size1()) {
if (IsLMDof(i_dof)) {
if (r_node.Is(ACTIVE)) {
mLMActiveIndices[lm_active_counter] = global_pos;
mGlobalToLocalIndexing[global_pos] = lm_active_counter;
mWhichBlockType[global_pos] = BlockType::LM_ACTIVE;
++lm_active_counter;
} else {
mLMInactiveIndices[lm_inactive_counter] = global_pos;
mGlobalToLocalIndexing[global_pos] = lm_inactive_counter;
mWhichBlockType[global_pos] = BlockType::LM_INACTIVE;
++lm_inactive_counter;
}
} else if ( r_node.Is(INTERFACE) && IsDisplacementDof(i_dof)) {
if (r_node.Is(MASTER)) {
mMasterIndices[master_counter] = global_pos;
mGlobalToLocalIndexing[global_pos] = master_counter;
mWhichBlockType[global_pos] = BlockType::MASTER;
++master_counter;
} else if (r_node.Is(SLAVE)) {
if (r_node.Is(ACTIVE)) {
mSlaveActiveIndices[slave_active_counter] = global_pos;
mGlobalToLocalIndexing[global_pos] = slave_active_counter;
mWhichBlockType[global_pos] = BlockType::SLAVE_ACTIVE;
++slave_active_counter;
} else {
mSlaveInactiveIndices[slave_inactive_counter] = global_pos;
mGlobalToLocalIndexing[global_pos] = slave_inactive_counter;
mWhichBlockType[global_pos] = BlockType::SLAVE_INACTIVE;
++slave_inactive_counter;
}
} else { 
mOtherIndices[other_counter] = global_pos;
mGlobalToLocalIndexing[global_pos] = other_counter;
mWhichBlockType[global_pos] = BlockType::OTHER;
++other_counter;
}
} else {
mOtherIndices[other_counter] = global_pos;
mGlobalToLocalIndexing[global_pos] = other_counter;
mWhichBlockType[global_pos] = BlockType::OTHER;
++other_counter;
}
++global_pos;
}
}
} else {
for (auto& i_dof : rDofSet) {
node_id = i_dof.Id();
const NodeType& r_node = rModelPart.GetNode(node_id);
if (IsLMDof(i_dof)) {
if (r_node.Is(ACTIVE)) {
mLMActiveIndices[lm_active_counter] = global_pos;
mGlobalToLocalIndexing[global_pos] = lm_active_counter;
mWhichBlockType[global_pos] = BlockType::LM_ACTIVE;
++lm_active_counter;
} else {
mLMInactiveIndices[lm_inactive_counter] = global_pos;
mGlobalToLocalIndexing[global_pos] = lm_inactive_counter;
mWhichBlockType[global_pos] = BlockType::LM_INACTIVE;
++lm_inactive_counter;
}
} else if ( r_node.Is(INTERFACE) && IsDisplacementDof(i_dof)) {
if (r_node.Is(MASTER)) {
mMasterIndices[master_counter] = global_pos;
mGlobalToLocalIndexing[global_pos] = master_counter;
mWhichBlockType[global_pos] = BlockType::MASTER;
++master_counter;
} else if (r_node.Is(SLAVE)) {
if (r_node.Is(ACTIVE)) {
mSlaveActiveIndices[slave_active_counter] = global_pos;
mGlobalToLocalIndexing[global_pos] = slave_active_counter;
mWhichBlockType[global_pos] = BlockType::SLAVE_ACTIVE;
++slave_active_counter;
} else {
mSlaveInactiveIndices[slave_inactive_counter] = global_pos;
mGlobalToLocalIndexing[global_pos] = slave_inactive_counter;
mWhichBlockType[global_pos] = BlockType::SLAVE_INACTIVE;
++slave_inactive_counter;
}
} else { 
mOtherIndices[other_counter] = global_pos;
mGlobalToLocalIndexing[global_pos] = other_counter;
mWhichBlockType[global_pos] = BlockType::OTHER;
++other_counter;
}
} else {
mOtherIndices[other_counter] = global_pos;
mGlobalToLocalIndexing[global_pos] = other_counter;
mWhichBlockType[global_pos] = BlockType::OTHER;
++other_counter;
}
++global_pos;
}
}

KRATOS_DEBUG_ERROR_IF(master_counter != n_master_dofs) << "The number of active slave dofs counter : " << master_counter << "is higher than the expected: " << n_master_dofs << std::endl;
KRATOS_DEBUG_ERROR_IF(slave_active_counter != n_slave_active_dofs) << "The number of active slave dofs counter : " << slave_active_counter << "is higher than the expected: " << n_slave_active_dofs << std::endl;
KRATOS_DEBUG_ERROR_IF(slave_inactive_counter != n_slave_inactive_dofs) << "The number of inactive slave dofs counter : " << slave_inactive_counter << "is higher than the expected: " << n_slave_inactive_dofs << std::endl;
KRATOS_DEBUG_ERROR_IF(lm_active_counter != n_lm_active_dofs) << "The number of active LM dofs counter : " << lm_active_counter << "is higher than the expected: " << n_lm_active_dofs << std::endl;
KRATOS_DEBUG_ERROR_IF(lm_inactive_counter != n_lm_inactive_dofs) << "The number of inactive LM dofs counter : " << lm_inactive_counter << "is higher than the expected: " << n_lm_inactive_dofs << std::endl;
KRATOS_DEBUG_ERROR_IF(other_counter != n_other_dofs) << "The number of other dofs counter : " << other_counter << "is higher than the expected: " << n_other_dofs << std::endl;

const auto it_dof_begin = rDofSet.begin();
mDisplacementDofs.reserve(mOtherIndices.size() + mMasterIndices.size() + mSlaveActiveIndices.size() + mSlaveInactiveIndices.size() + mLMInactiveIndices.size() + mLMActiveIndices.size());

std::size_t counter = 0;
for (auto& r_index : mOtherIndices) {
auto it_dof = it_dof_begin + r_index;
auto* p_dof = new DofType(*it_dof);
p_dof->SetEquationId(counter);
mDisplacementDofs.push_back(p_dof);
++counter;
}
for (auto& r_index : mMasterIndices) {
auto it_dof = it_dof_begin + r_index;
auto* p_dof = new DofType(*it_dof);
p_dof->SetEquationId(counter);
mDisplacementDofs.push_back(p_dof);
++counter;
}
for (auto& r_index : mSlaveInactiveIndices) {
auto it_dof = it_dof_begin + r_index;
auto* p_dof = new DofType(*it_dof);
p_dof->SetEquationId(counter);
mDisplacementDofs.push_back(p_dof);
++counter;
}
for (auto& r_index : mSlaveActiveIndices) {
auto it_dof = it_dof_begin + r_index;
auto* p_dof = new DofType(*it_dof);
p_dof->SetEquationId(counter);
mDisplacementDofs.push_back(p_dof);
++counter;
}

if(mpSolverDispBlock->AdditionalPhysicalDataIsNeeded() ) {
mpSolverDispBlock->ProvideAdditionalData(rA, rX, rB, mDisplacementDofs, rModelPart);
}
}



DofsArrayType& GetDisplacementDofs()
{
return mDisplacementDofs;
}


const DofsArrayType& GetDisplacementDofs() const 
{
return mDisplacementDofs;
}



std::string Info() const override
{
return "Mixed displacement LM linear solver";
}

void PrintInfo (std::ostream& rOStream) const override
{
rOStream << "Mixed displacement LM linear solver";
}

void PrintData (std::ostream& rOStream) const override
{
}


protected:





void FillBlockMatrices (
const bool NeedAllocation,
SparseMatrixType& rA,
VectorType& rX,
VectorType& rB
)
{
KRATOS_TRY

const SizeType other_dof_size = mOtherIndices.size();
const SizeType master_size = mMasterIndices.size();
const SizeType slave_inactive_size = mSlaveInactiveIndices.size();
const SizeType slave_active_size = mSlaveActiveIndices.size();
const SizeType lm_active_size = mLMActiveIndices.size();
const SizeType lm_inactive_size = mLMInactiveIndices.size();

if (NeedAllocation)
AllocateBlocks();

const IndexType* index1 = rA.index1_data().begin();
const IndexType* index2 = rA.index2_data().begin();
const double* values = rA.value_data().begin();

SparseMatrixType KMLMA(master_size, lm_active_size);            
SparseMatrixType KLMALMA(lm_active_size, lm_active_size);       
SparseMatrixType KSALMA(slave_active_size, lm_active_size);     
SparseMatrixType KLMILMI(lm_inactive_size, lm_inactive_size);   

IndexType* KMLMA_ptr = new IndexType[master_size + 1];
IndexType* mKSAN_ptr = new IndexType[slave_active_size + 1];
IndexType* mKSAM_ptr = new IndexType[slave_active_size + 1];
IndexType* mKSASI_ptr = new IndexType[slave_active_size + 1];
IndexType* mKSASA_ptr = new IndexType[slave_active_size + 1];
IndexType* KSALMA_ptr = new IndexType[slave_active_size + 1];
IndexType* KLMILMI_ptr = new IndexType[lm_inactive_size + 1];
IndexType* KLMALMA_ptr = new IndexType[lm_active_size + 1];

IndexPartition<std::size_t>(master_size +1).for_each([&](std::size_t i) {
KMLMA_ptr[i] = 0;
});
IndexPartition<std::size_t>(slave_active_size +1).for_each([&](std::size_t i) {
mKSAN_ptr[i] = 0;
mKSAM_ptr[i] = 0;
mKSASI_ptr[i] = 0;
mKSASA_ptr[i] = 0;
KSALMA_ptr[i] = 0;
});
IndexPartition<std::size_t>(lm_inactive_size +1).for_each([&](std::size_t i) {
KLMILMI_ptr[i] = 0;
});
IndexPartition<std::size_t>(lm_active_size +1).for_each([&](std::size_t i) {
KLMALMA_ptr[i] = 0;
});

IndexPartition<std::size_t>(rA.size1()).for_each([&](std::size_t i) {
const IndexType row_begin = index1[i];
const IndexType row_end   = index1[i+1];
const IndexType local_row_id = mGlobalToLocalIndexing[i];

IndexType KMLMA_cols = 0;
IndexType mKSAN_cols = 0;
IndexType mKSAM_cols = 0;
IndexType mKSASI_cols = 0;
IndexType mKSASA_cols = 0;
IndexType KSALMA_cols = 0;
IndexType KLMILMI_cols = 0;
IndexType KLMALMA_cols = 0;

if ( mWhichBlockType[i] == BlockType::MASTER) { 
for (IndexType j=row_begin; j<row_end; j++) {
const IndexType col_index = index2[j];
if ( mWhichBlockType[col_index] == BlockType::LM_ACTIVE) { 
++KMLMA_cols;
}
}
KRATOS_DEBUG_ERROR_IF(local_row_id > master_size) << "MASTER:: Local row ID: " << local_row_id <<" is greater than the number of rows " << master_size << std::endl;
KMLMA_ptr[local_row_id + 1] = KMLMA_cols;
} else if ( mWhichBlockType[i] == BlockType::SLAVE_ACTIVE) { 
for (IndexType j=row_begin; j<row_end; j++) {
const IndexType col_index = index2[j];
if (mWhichBlockType[col_index] == BlockType::OTHER) {                 
++mKSAN_cols;
} else if (mWhichBlockType[col_index] == BlockType::MASTER) {         
++mKSAM_cols;
} else if (mWhichBlockType[col_index] == BlockType::SLAVE_INACTIVE) { 
++mKSASI_cols;
} else if (mWhichBlockType[col_index] == BlockType::SLAVE_ACTIVE) {   
++mKSASA_cols;
} else if ( mWhichBlockType[col_index] == BlockType::LM_ACTIVE) {     
++KSALMA_cols;
}
}
KRATOS_DEBUG_ERROR_IF(local_row_id > slave_active_size) << "SLAVE_ACTIVE:: Local row ID: " << local_row_id <<" is greater than the number of rows " << slave_active_size << std::endl;
mKSAN_ptr[local_row_id + 1]  = mKSAN_cols;
mKSAM_ptr[local_row_id + 1]  = mKSAM_cols;
mKSASI_ptr[local_row_id + 1] = mKSASI_cols;
mKSASA_ptr[local_row_id + 1] = mKSASA_cols;
KSALMA_ptr[local_row_id + 1] = KSALMA_cols;
} else if ( mWhichBlockType[i] == BlockType::LM_INACTIVE) { 
for (IndexType j=row_begin; j<row_end; j++) {
const IndexType col_index = index2[j];
if (mWhichBlockType[col_index] == BlockType::LM_INACTIVE) { 
++KLMILMI_cols;
}
}
KRATOS_DEBUG_ERROR_IF(local_row_id > lm_inactive_size) << "LM_INACTIVE:: Local row ID: " << local_row_id <<" is greater than the number of rows " << lm_inactive_size << std::endl;
KLMILMI_ptr[local_row_id + 1] = KLMILMI_cols;
} else if ( mWhichBlockType[i] == BlockType::LM_ACTIVE) { 
for (IndexType j=row_begin; j<row_end; j++) {
const IndexType col_index = index2[j];
if (mWhichBlockType[col_index] == BlockType::LM_ACTIVE) { 
++KLMALMA_cols;
}
}
KRATOS_DEBUG_ERROR_IF(local_row_id > lm_active_size) << "LM_ACTIVE:: Local row ID: " << local_row_id <<" is greater than the number of rows " << lm_active_size << std::endl;
KLMALMA_ptr[local_row_id + 1] = KLMALMA_cols;
}
});

std::partial_sum(KMLMA_ptr, KMLMA_ptr + master_size + 1, KMLMA_ptr);
const std::size_t KMLMA_nonzero_values = KMLMA_ptr[master_size];
IndexType* aux_index2_KMLMA= new IndexType[KMLMA_nonzero_values];
double* aux_val_KMLMA= new double[KMLMA_nonzero_values];

std::partial_sum(mKSAN_ptr, mKSAN_ptr + slave_active_size + 1, mKSAN_ptr);
const std::size_t mKSAN_nonzero_values = mKSAN_ptr[slave_active_size];
IndexType* aux_index2_mKSAN= new IndexType[mKSAN_nonzero_values];
double* aux_val_mKSAN= new double[mKSAN_nonzero_values];

std::partial_sum(mKSAM_ptr, mKSAM_ptr + slave_active_size + 1, mKSAM_ptr);
const std::size_t mKSAM_nonzero_values = mKSAM_ptr[slave_active_size];
IndexType* aux_index2_mKSAM= new IndexType[mKSAM_nonzero_values];
double* aux_val_mKSAM= new double[mKSAM_nonzero_values];

std::partial_sum(mKSASI_ptr, mKSASI_ptr + slave_active_size + 1, mKSASI_ptr);
const std::size_t mKSASI_nonzero_values = mKSASI_ptr[slave_active_size];
IndexType* aux_index2_mKSASI= new IndexType[mKSASI_nonzero_values];
double* aux_val_mKSASI= new double[mKSASI_nonzero_values];

std::partial_sum(mKSASA_ptr, mKSASA_ptr + slave_active_size + 1, mKSASA_ptr);
const std::size_t mKSASA_nonzero_values = mKSASA_ptr[slave_active_size];
IndexType* aux_index2_mKSASA= new IndexType[mKSASA_nonzero_values];
double* aux_val_mKSASA = new double[mKSASA_nonzero_values];

std::partial_sum(KSALMA_ptr, KSALMA_ptr + slave_active_size + 1, KSALMA_ptr);
const std::size_t KSALMA_nonzero_values = KSALMA_ptr[slave_active_size];
IndexType* aux_index2_KSALMA= new IndexType[KSALMA_nonzero_values];
double* aux_val_KSALMA = new double[KSALMA_nonzero_values];

std::partial_sum(KLMILMI_ptr, KLMILMI_ptr + lm_inactive_size + 1, KLMILMI_ptr);
const std::size_t KLMILMI_nonzero_values = KLMILMI_ptr[lm_inactive_size];
IndexType* aux_index2_KLMILMI= new IndexType[KLMILMI_nonzero_values];
double* aux_val_KLMILMI = new double[KLMILMI_nonzero_values];

std::partial_sum(KLMALMA_ptr, KLMALMA_ptr + lm_active_size + 1, KLMALMA_ptr);
const std::size_t KLMALMA_nonzero_values = KLMALMA_ptr[lm_active_size];
IndexType* aux_index2_KLMALMA = new IndexType[KLMALMA_nonzero_values];
double* aux_val_KLMALMA = new double[KLMALMA_nonzero_values];

IndexPartition<std::size_t>(rA.size1()).for_each([&](std::size_t i) {
const IndexType row_begin = index1[i];
const IndexType row_end   = index1[i+1];
const IndexType local_row_id = mGlobalToLocalIndexing[i];

if ( mWhichBlockType[i] == BlockType::MASTER) { 
IndexType KMLMA_row_beg = KMLMA_ptr[local_row_id];
IndexType KMLMA_row_end = KMLMA_row_beg;
for (IndexType j=row_begin; j<row_end; j++) {
const IndexType col_index = index2[j];
if ( mWhichBlockType[col_index] == BlockType::LM_ACTIVE) { 
const double value = values[j];
const IndexType local_col_id = mGlobalToLocalIndexing[col_index];
aux_index2_KMLMA[KMLMA_row_end] = local_col_id;
aux_val_KMLMA[KMLMA_row_end] = value;
++KMLMA_row_end;
}
}
} else if ( mWhichBlockType[i] == BlockType::SLAVE_ACTIVE) { 
IndexType mKSAN_row_beg = mKSAN_ptr[local_row_id];
IndexType mKSAN_row_end = mKSAN_row_beg;
IndexType mKSAM_row_beg = mKSAM_ptr[local_row_id];
IndexType mKSAM_row_end = mKSAM_row_beg;
IndexType mKSASI_row_beg = mKSASI_ptr[local_row_id];
IndexType mKSASI_row_end = mKSASI_row_beg;
IndexType mKSASA_row_beg = mKSASA_ptr[local_row_id];
IndexType mKSASA_row_end = mKSASA_row_beg;
IndexType KSALMA_row_beg = KSALMA_ptr[local_row_id];
IndexType KSALMA_row_end = KSALMA_row_beg;
for (IndexType j=row_begin; j<row_end; j++) {
const IndexType col_index = index2[j];
const double value = values[j];
const IndexType local_col_id = mGlobalToLocalIndexing[col_index];
if (mWhichBlockType[col_index] == BlockType::OTHER) {                 
aux_index2_mKSAN[mKSAN_row_end] = local_col_id;
aux_val_mKSAN[mKSAN_row_end] = value;
++mKSAN_row_end;
} else if (mWhichBlockType[col_index] == BlockType::MASTER) {         
aux_index2_mKSAM[mKSAM_row_end] = local_col_id;
aux_val_mKSAM[mKSAM_row_end] = value;
++mKSAM_row_end;
} else if (mWhichBlockType[col_index] == BlockType::SLAVE_INACTIVE) { 
aux_index2_mKSASI[mKSASI_row_end] = local_col_id;
aux_val_mKSASI[mKSASI_row_end] = value;
++mKSASI_row_end;
} else if (mWhichBlockType[col_index] == BlockType::SLAVE_ACTIVE) {   
aux_index2_mKSASA[mKSASA_row_end] = local_col_id;
aux_val_mKSASA[mKSASA_row_end] = value;
++mKSASA_row_end;
} else if ( mWhichBlockType[col_index] == BlockType::LM_ACTIVE) {     
aux_index2_KSALMA[KSALMA_row_end] = local_col_id;
aux_val_KSALMA[KSALMA_row_end] = value;
++KSALMA_row_end;
}
}
} else if ( mWhichBlockType[i] == BlockType::LM_INACTIVE) { 
IndexType KLMILMI_row_beg = KLMILMI_ptr[local_row_id];
IndexType KLMILMI_row_end = KLMILMI_row_beg;
for (IndexType j=row_begin; j<row_end; j++) {
const IndexType col_index = index2[j];
if (mWhichBlockType[col_index] == BlockType::LM_INACTIVE) { 
const double value = values[j];
const IndexType local_col_id = mGlobalToLocalIndexing[col_index];
aux_index2_KLMILMI[KLMILMI_row_end] = local_col_id;
aux_val_KLMILMI[KLMILMI_row_end] = value;
++KLMILMI_row_end;
}
}
} else if ( mWhichBlockType[i] == BlockType::LM_ACTIVE) { 
IndexType KLMALMA_row_beg = KLMALMA_ptr[local_row_id];
IndexType KLMALMA_row_end = KLMALMA_row_beg;
for (IndexType j=row_begin; j<row_end; j++) {
const IndexType col_index = index2[j];
if (mWhichBlockType[col_index] == BlockType::LM_ACTIVE) { 
const double value = values[j];
const IndexType local_col_id = mGlobalToLocalIndexing[col_index];
aux_index2_KLMALMA[KLMALMA_row_end] = local_col_id;
aux_val_KLMALMA[KLMALMA_row_end] = value;
++KLMALMA_row_end;
}
}
}
});

CreateMatrix(KMLMA, master_size, lm_active_size, KMLMA_ptr, aux_index2_KMLMA, aux_val_KMLMA);
CreateMatrix(mKSAN, slave_active_size, other_dof_size, mKSAN_ptr, aux_index2_mKSAN, aux_val_mKSAN);
CreateMatrix(mKSAM, slave_active_size, master_size, mKSAM_ptr, aux_index2_mKSAM, aux_val_mKSAM);
CreateMatrix(mKSASI, slave_active_size, slave_inactive_size, mKSASI_ptr, aux_index2_mKSASI, aux_val_mKSASI);
CreateMatrix(mKSASA, slave_active_size, slave_active_size, mKSASA_ptr, aux_index2_mKSASA, aux_val_mKSASA);
CreateMatrix(KSALMA, slave_active_size, lm_active_size, KSALMA_ptr, aux_index2_KSALMA, aux_val_KSALMA);
CreateMatrix(KLMILMI, lm_inactive_size, lm_inactive_size, KLMILMI_ptr, aux_index2_KLMILMI, aux_val_KLMILMI);
CreateMatrix(KLMALMA, lm_active_size, lm_active_size, KLMALMA_ptr, aux_index2_KLMALMA, aux_val_KLMALMA);

if (lm_active_size > 0) {
ComputeDiagonalByLumping(KSALMA, mKLMAModified, ZeroTolerance);
}

if (lm_inactive_size > 0) {
ComputeDiagonalByLumping(KLMILMI, mKLMIModified, ZeroTolerance);
}

if (slave_active_size > 0) {
SparseMatrixMultiplicationUtility::MatrixMultiplication(KMLMA,   mKLMAModified, mPOperator);
SparseMatrixMultiplicationUtility::MatrixMultiplication(KLMALMA, mKLMAModified, mCOperator);
}

SparseMatrixType master_auxKSAN(master_size, other_dof_size);
SparseMatrixType master_auxKSAM(master_size, master_size);
SparseMatrixType master_auxKSASI(master_size, slave_inactive_size);
SparseMatrixType master_auxKSASA(master_size, slave_active_size);

if (slave_active_size > 0) {
SparseMatrixMultiplicationUtility::MatrixMultiplication(mPOperator, mKSAN, master_auxKSAN);
SparseMatrixMultiplicationUtility::MatrixMultiplication(mPOperator, mKSAM, master_auxKSAM);
if (slave_inactive_size > 0)
SparseMatrixMultiplicationUtility::MatrixMultiplication(mPOperator, mKSASI, master_auxKSASI);
SparseMatrixMultiplicationUtility::MatrixMultiplication(mPOperator, mKSASA, master_auxKSASA);
}

SparseMatrixType aslave_auxKSAN(slave_active_size, other_dof_size);
SparseMatrixType aslave_auxKSAM(slave_active_size, master_size);
SparseMatrixType aslave_auxKSASI(slave_active_size, slave_inactive_size);
SparseMatrixType aslave_auxKSASA(slave_active_size, slave_active_size);

if (slave_active_size > 0) {
SparseMatrixMultiplicationUtility::MatrixMultiplication(mCOperator, mKSAN, aslave_auxKSAN);
SparseMatrixMultiplicationUtility::MatrixMultiplication(mCOperator, mKSAM, aslave_auxKSAM);
if (slave_inactive_size > 0)
SparseMatrixMultiplicationUtility::MatrixMultiplication(mCOperator, mKSASI, aslave_auxKSASI);
SparseMatrixMultiplicationUtility::MatrixMultiplication(mCOperator, mKSASA, aslave_auxKSASA);
}

const SizeType other_dof_initial_index = 0;
const SizeType master_dof_initial_index = other_dof_size;
const SizeType slave_inactive_dof_initial_index = master_dof_initial_index + master_size;
const SizeType assembling_slave_dof_initial_index = slave_inactive_dof_initial_index + slave_inactive_size;

const SizeType nrows = mKDispModified.size1();
const SizeType ncols = mKDispModified.size2();
IndexType* K_disp_modified_ptr_aux1 = new IndexType[nrows + 1];
K_disp_modified_ptr_aux1[0] = 0;

IndexPartition<std::size_t>(rA.size1()).for_each([&](std::size_t i) {
if ( mWhichBlockType[i] == BlockType::OTHER) { 
ComputeNonZeroColumnsDispDoFs( index1, index2, values,  i, other_dof_initial_index, K_disp_modified_ptr_aux1);
} else if ( mWhichBlockType[i] == BlockType::MASTER) { 
ComputeNonZeroColumnsDispDoFs( index1, index2, values,  i, master_dof_initial_index, K_disp_modified_ptr_aux1);
} else if ( mWhichBlockType[i] == BlockType::SLAVE_INACTIVE) { 
ComputeNonZeroColumnsDispDoFs( index1, index2, values,  i, slave_inactive_dof_initial_index, K_disp_modified_ptr_aux1);
} else if ( mWhichBlockType[i] == BlockType::LM_ACTIVE) { 
ComputeNonZeroColumnsPartialDispDoFs( index1, index2, values,  i, assembling_slave_dof_initial_index, K_disp_modified_ptr_aux1);
}
});

std::partial_sum(K_disp_modified_ptr_aux1, K_disp_modified_ptr_aux1 + nrows + 1, K_disp_modified_ptr_aux1);
const SizeType nonzero_values_aux1 = K_disp_modified_ptr_aux1[nrows];
IndexType* aux_index2_K_disp_modified_aux1 = new IndexType[nonzero_values_aux1];
double* aux_val_K_disp_modified_aux1 = new double[nonzero_values_aux1];

IndexPartition<std::size_t>(rA.size1()).for_each([&](std::size_t i) {
if ( mWhichBlockType[i] == BlockType::OTHER) { 
ComputeAuxiliaryValuesDispDoFs( index1, index2, values,  i, other_dof_initial_index, K_disp_modified_ptr_aux1, aux_index2_K_disp_modified_aux1, aux_val_K_disp_modified_aux1);
} else if ( mWhichBlockType[i] == BlockType::MASTER) { 
ComputeAuxiliaryValuesDispDoFs( index1, index2, values,  i, master_dof_initial_index, K_disp_modified_ptr_aux1, aux_index2_K_disp_modified_aux1, aux_val_K_disp_modified_aux1);
} else if ( mWhichBlockType[i] == BlockType::SLAVE_INACTIVE) { 
ComputeAuxiliaryValuesDispDoFs( index1, index2, values,  i, slave_inactive_dof_initial_index, K_disp_modified_ptr_aux1, aux_index2_K_disp_modified_aux1, aux_val_K_disp_modified_aux1);
} else if ( mWhichBlockType[i] == BlockType::LM_ACTIVE) { 
ComputeAuxiliaryValuesPartialDispDoFs( index1, index2, values,  i, assembling_slave_dof_initial_index, K_disp_modified_ptr_aux1, aux_index2_K_disp_modified_aux1, aux_val_K_disp_modified_aux1);
}
});

CreateMatrix(mKDispModified, nrows, ncols, K_disp_modified_ptr_aux1, aux_index2_K_disp_modified_aux1, aux_val_K_disp_modified_aux1);

IndexType* K_disp_modified_ptr_aux2 = new IndexType[nrows + 1];
IndexPartition<std::size_t>(nrows + 1).for_each([&](std::size_t i) {
K_disp_modified_ptr_aux2[i] = 0;
});

IndexPartition<std::size_t>(master_size).for_each([&](std::size_t i) {
IndexType K_disp_modified_cols_aux2 = 0;
if (master_auxKSAN.nnz() > 0 && other_dof_size > 0) {
SparseMatrixMultiplicationUtility::ComputeNonZeroBlocks(master_auxKSAN, i, K_disp_modified_cols_aux2);
}
if (master_auxKSAM.nnz() > 0) {
SparseMatrixMultiplicationUtility::ComputeNonZeroBlocks(master_auxKSAM, i, K_disp_modified_cols_aux2);
}
if (master_auxKSASI.nnz() > 0 && slave_inactive_size > 0) {
SparseMatrixMultiplicationUtility::ComputeNonZeroBlocks(master_auxKSASI, i, K_disp_modified_cols_aux2);
}
if (master_auxKSASA.nnz() > 0 && slave_active_size > 0) {
SparseMatrixMultiplicationUtility::ComputeNonZeroBlocks(master_auxKSASA, i, K_disp_modified_cols_aux2);
}
K_disp_modified_ptr_aux2[master_dof_initial_index + i + 1] = K_disp_modified_cols_aux2;
});

IndexPartition<std::size_t>(slave_active_size).for_each([&](std::size_t i) {
IndexType K_disp_modified_cols_aux2 = 0;
if (aslave_auxKSAN.nnz() > 0 && other_dof_size > 0) {
SparseMatrixMultiplicationUtility::ComputeNonZeroBlocks(aslave_auxKSAN, i, K_disp_modified_cols_aux2);
}
if (aslave_auxKSAM.nnz() > 0 && master_size > 0) {
SparseMatrixMultiplicationUtility::ComputeNonZeroBlocks(aslave_auxKSAM, i, K_disp_modified_cols_aux2);
}
if (aslave_auxKSASI.nnz() > 0 && slave_inactive_size > 0) {
SparseMatrixMultiplicationUtility::ComputeNonZeroBlocks(aslave_auxKSASI, i, K_disp_modified_cols_aux2);
}
if (aslave_auxKSASA.nnz() > 0) {
SparseMatrixMultiplicationUtility::ComputeNonZeroBlocks(aslave_auxKSASA, i, K_disp_modified_cols_aux2);
}
K_disp_modified_ptr_aux2[assembling_slave_dof_initial_index + i + 1] = K_disp_modified_cols_aux2;
});

std::partial_sum(K_disp_modified_ptr_aux2, K_disp_modified_ptr_aux2 + nrows + 1, K_disp_modified_ptr_aux2);
const SizeType nonzero_values_aux2 = K_disp_modified_ptr_aux2[nrows];
IndexType* aux_index2_K_disp_modified_aux2 = new IndexType[nonzero_values_aux2];
double* aux_val_K_disp_modified_aux2 = new double[nonzero_values_aux2];

IndexPartition<std::size_t>(master_size).for_each([&](std::size_t i) {
const IndexType row_beg = K_disp_modified_ptr_aux2[master_dof_initial_index + i];
IndexType row_end = row_beg;
if (master_auxKSAN.nnz() > 0 && other_dof_size > 0) {
SparseMatrixMultiplicationUtility::ComputeAuxiliarValuesBlocks(master_auxKSAN, aux_index2_K_disp_modified_aux2, aux_val_K_disp_modified_aux2, i, row_end, other_dof_initial_index);
}
if (master_auxKSAM.nnz() > 0) {
SparseMatrixMultiplicationUtility::ComputeAuxiliarValuesBlocks(master_auxKSAM, aux_index2_K_disp_modified_aux2, aux_val_K_disp_modified_aux2, i, row_end, master_dof_initial_index);
}
if (master_auxKSASI.nnz() > 0 && slave_inactive_size > 0) {
SparseMatrixMultiplicationUtility::ComputeAuxiliarValuesBlocks(master_auxKSASI, aux_index2_K_disp_modified_aux2, aux_val_K_disp_modified_aux2, i, row_end, slave_inactive_dof_initial_index);
}
if (master_auxKSASA.nnz() > 0 && slave_active_size > 0) {
SparseMatrixMultiplicationUtility::ComputeAuxiliarValuesBlocks(master_auxKSASA, aux_index2_K_disp_modified_aux2, aux_val_K_disp_modified_aux2, i, row_end, assembling_slave_dof_initial_index);
}
});

IndexPartition<std::size_t>(slave_active_size).for_each([&](std::size_t i) {
const IndexType row_beg = K_disp_modified_ptr_aux2[assembling_slave_dof_initial_index + i];
IndexType row_end = row_beg;
if (aslave_auxKSAN.nnz() > 0 && other_dof_size > 0) {
SparseMatrixMultiplicationUtility::ComputeAuxiliarValuesBlocks(aslave_auxKSAN, aux_index2_K_disp_modified_aux2, aux_val_K_disp_modified_aux2, i, row_end, other_dof_initial_index);
}
if (aslave_auxKSAM.nnz() > 0 && master_size > 0) {
SparseMatrixMultiplicationUtility::ComputeAuxiliarValuesBlocks(aslave_auxKSAM, aux_index2_K_disp_modified_aux2, aux_val_K_disp_modified_aux2, i, row_end, master_dof_initial_index);
}
if (aslave_auxKSASI.nnz() > 0 && slave_inactive_size > 0) {
SparseMatrixMultiplicationUtility::ComputeAuxiliarValuesBlocks(aslave_auxKSASI, aux_index2_K_disp_modified_aux2, aux_val_K_disp_modified_aux2, i, row_end, slave_inactive_dof_initial_index);
}
if (aslave_auxKSASA.nnz() > 0) {
SparseMatrixMultiplicationUtility::ComputeAuxiliarValuesBlocks(aslave_auxKSASA, aux_index2_K_disp_modified_aux2, aux_val_K_disp_modified_aux2, i, row_end, assembling_slave_dof_initial_index);
}
});

SparseMatrixType K_disp_modified_aux2(nrows, ncols);
CreateMatrix(K_disp_modified_aux2, nrows, ncols, K_disp_modified_ptr_aux2, aux_index2_K_disp_modified_aux2, aux_val_K_disp_modified_aux2);

SparseMatrixMultiplicationUtility::MatrixAdd<SparseMatrixType, SparseMatrixType>(mKDispModified, K_disp_modified_aux2, - 1.0);

EnsureStructuralSymmetryMatrix(mKDispModified);

#ifdef KRATOS_DEBUG
CheckMatrix(mKDispModified);
#endif


KRATOS_CATCH ("")
}

private:

LinearSolverPointerType mpSolverDispBlock; 

Flags mOptions; 

DofsArrayType mDisplacementDofs; 

IndexVectorType mMasterIndices;         
IndexVectorType mSlaveInactiveIndices;  
IndexVectorType mSlaveActiveIndices;    
IndexVectorType mLMInactiveIndices;     
IndexVectorType mLMActiveIndices;       
IndexVectorType mOtherIndices;          
IndexVectorType mGlobalToLocalIndexing; 
BlockTypeVectorType mWhichBlockType; 

SparseMatrixType mKDispModified; 
SparseMatrixType mKLMAModified;  
SparseMatrixType mKLMIModified;  

SparseMatrixType mKSAN;    
SparseMatrixType mKSAM;    
SparseMatrixType mKSASI;   
SparseMatrixType mKSASA;   

SparseMatrixType mPOperator; 
SparseMatrixType mCOperator; 

VectorType mResidualLMActive;   
VectorType mResidualLMInactive; 
VectorType mResidualDisp;       

VectorType mLMActive;           
VectorType mLMInactive;         
VectorType mDisp;               

IndexType mEchoLevel = 0;       
IndexType mFileCreated = 0;     




inline void ComputeNonZeroColumnsDispDoFs(
const IndexType* Index1,
const IndexType* Index2,
const double* Values,
const int CurrentRow,
const IndexType InitialIndex,
IndexType* Ptr
)
{
const IndexType row_begin = Index1[CurrentRow];
const IndexType row_end   = Index1[CurrentRow + 1];

IndexType cols = 0;

const IndexType local_row_id = mGlobalToLocalIndexing[CurrentRow] + InitialIndex;
for (IndexType j=row_begin; j<row_end; j++) {
const IndexType col_index = Index2[j];
if (mWhichBlockType[col_index] == BlockType::OTHER) {
++cols;
} else if (mWhichBlockType[col_index] == BlockType::MASTER) {
++cols;
} else if (mWhichBlockType[col_index] == BlockType::SLAVE_INACTIVE) {
++cols;
} else if (mWhichBlockType[col_index] == BlockType::SLAVE_ACTIVE) {
++cols;
}
}
Ptr[local_row_id + 1] = cols;
}


inline void ComputeNonZeroColumnsPartialDispDoFs(
const IndexType* Index1,
const IndexType* Index2,
const double* Values,
const int CurrentRow,
const IndexType InitialIndex,
IndexType* Ptr
)
{
const IndexType row_begin = Index1[CurrentRow];
const IndexType row_end   = Index1[CurrentRow + 1];

IndexType cols = 0;

const IndexType local_row_id = mGlobalToLocalIndexing[CurrentRow] + InitialIndex;
for (IndexType j=row_begin; j<row_end; j++) {
const IndexType col_index = Index2[j];
if (mWhichBlockType[col_index] == BlockType::MASTER) {
++cols;
} else if (mWhichBlockType[col_index] == BlockType::SLAVE_INACTIVE) {
++cols;
} else if (mWhichBlockType[col_index] == BlockType::SLAVE_ACTIVE) {
++cols;
}
}
Ptr[local_row_id + 1] = cols;
}


inline void ComputeAuxiliaryValuesDispDoFs(
const IndexType* Index1,
const IndexType* Index2,
const double* Values,
const int CurrentRow,
const IndexType InitialIndex,
IndexType* Ptr,
IndexType* AuxIndex2,
double* AuxVals
)
{
const SizeType other_dof_size = mOtherIndices.size();
const SizeType master_size = mMasterIndices.size();
const SizeType slave_inactive_size = mSlaveInactiveIndices.size();

const SizeType other_dof_initial_index = 0;
const SizeType master_dof_initial_index = other_dof_size;
const SizeType slave_inactive_dof_initial_index = master_dof_initial_index + master_size;
const SizeType assembling_slave_dof_initial_index = slave_inactive_dof_initial_index + slave_inactive_size;

const IndexType local_row_id = mGlobalToLocalIndexing[CurrentRow] + InitialIndex;

const IndexType row_begin_A = Index1[CurrentRow];
const IndexType row_end_A   = Index1[CurrentRow + 1];

const IndexType row_beg = Ptr[local_row_id];
IndexType row_end = row_beg;

for (IndexType j=row_begin_A; j<row_end_A; j++) {
const IndexType col_index = Index2[j];
const IndexType local_col_id = mGlobalToLocalIndexing[col_index];
const double value = Values[j];
if (mWhichBlockType[col_index] == BlockType::OTHER) {
AuxIndex2[row_end] = local_col_id + other_dof_initial_index;
AuxVals[row_end] = value;
++row_end;
} else if (mWhichBlockType[col_index] == BlockType::MASTER) {
AuxIndex2[row_end] = local_col_id + master_dof_initial_index;
AuxVals[row_end] = value;
++row_end;
} else if (mWhichBlockType[col_index] == BlockType::SLAVE_INACTIVE) {
AuxIndex2[row_end] = local_col_id + slave_inactive_dof_initial_index;
AuxVals[row_end] = value;
++row_end;
} else if (mWhichBlockType[col_index] == BlockType::SLAVE_ACTIVE) {
AuxIndex2[row_end] = local_col_id + assembling_slave_dof_initial_index;
AuxVals[row_end] = value;
++row_end;
}
}
}


inline void ComputeAuxiliaryValuesPartialDispDoFs(
const IndexType* Index1,
const IndexType* Index2,
const double* Values,
const int CurrentRow,
const IndexType InitialIndex,
IndexType* Ptr,
IndexType* AuxIndex2,
double* AuxVals
)
{
const SizeType other_dof_size = mOtherIndices.size();
const SizeType master_size = mMasterIndices.size();
const SizeType slave_inactive_size = mSlaveInactiveIndices.size();

const SizeType master_dof_initial_index = other_dof_size;
const SizeType slave_inactive_dof_initial_index = master_dof_initial_index + master_size;
const SizeType assembling_slave_dof_initial_index = slave_inactive_dof_initial_index + slave_inactive_size;

const IndexType local_row_id = mGlobalToLocalIndexing[CurrentRow] + InitialIndex;

const IndexType row_begin_A = Index1[CurrentRow];
const IndexType row_end_A   = Index1[CurrentRow + 1];

const IndexType row_beg = Ptr[local_row_id];
IndexType row_end = row_beg;

for (IndexType j=row_begin_A; j<row_end_A; j++) {
const IndexType col_index = Index2[j];
const IndexType local_col_id = mGlobalToLocalIndexing[col_index];
const double value = Values[j];
if (mWhichBlockType[col_index] == BlockType::MASTER) {
AuxIndex2[row_end] = local_col_id + master_dof_initial_index;
AuxVals[row_end] = value;
++row_end;
} else if (mWhichBlockType[col_index] == BlockType::SLAVE_INACTIVE) {
AuxIndex2[row_end] = local_col_id + slave_inactive_dof_initial_index;
AuxVals[row_end] = value;
++row_end;
} else if (mWhichBlockType[col_index] == BlockType::SLAVE_ACTIVE) {
AuxIndex2[row_end] = local_col_id + assembling_slave_dof_initial_index;
AuxVals[row_end] = value;
++row_end;
}
}
}


inline void AllocateBlocks()
{
auto& r_data_dofs = mDisplacementDofs.GetContainer(); 
for (IndexType i=0; i<r_data_dofs.size(); ++i) {
delete r_data_dofs[i];
}
r_data_dofs.clear();

mKDispModified.clear(); 
mKLMAModified.clear();  
mKLMIModified.clear();  

mKSAN.clear();  
mKSAM.clear();  
mKSASI.clear(); 
mKSASA.clear(); 

mPOperator.clear(); 
mCOperator.clear(); 

mResidualLMActive.clear();   
mResidualLMInactive.clear(); 
mResidualDisp.clear();       

mLMActive.clear();   
mLMInactive.clear(); 
mDisp.clear();       

const SizeType other_dof_size = mOtherIndices.size();
const SizeType master_size = mMasterIndices.size();
const SizeType slave_inactive_size = mSlaveInactiveIndices.size();
const SizeType slave_active_size = mSlaveActiveIndices.size();
const SizeType lm_active_size = mLMActiveIndices.size();
const SizeType lm_inactive_size = mLMInactiveIndices.size();
const SizeType total_size = other_dof_size + master_size + slave_inactive_size + slave_active_size;

mKDispModified.resize(total_size, total_size, false);            
mKLMAModified.resize(lm_active_size, lm_active_size, false);     
mKLMAModified.reserve(lm_active_size);
mKLMIModified.resize(lm_inactive_size, lm_inactive_size, false); 
mKLMIModified.reserve(lm_inactive_size);

mKSAN.resize(slave_active_size, other_dof_size, false);       
mKSAM.resize(slave_active_size, master_size, false);          
mKSASI.resize(slave_active_size, slave_inactive_size, false); 
mKSASA.resize(slave_active_size, slave_active_size, false);   

mPOperator.resize(master_size, slave_active_size, false);    
mCOperator.resize(lm_active_size, slave_active_size, false); 

mResidualLMActive.resize(lm_active_size, false );     
mResidualLMInactive.resize(lm_inactive_size, false ); 
mResidualDisp.resize(total_size );             

mLMActive.resize(lm_active_size, false);     
mLMInactive.resize(lm_inactive_size, false); 
mDisp.resize(total_size, false);             
}


inline void GetUPart (
const VectorType& rTotalResidual,
VectorType& ResidualU
)
{
const SizeType other_dof_size = mOtherIndices.size();
const SizeType master_size = mMasterIndices.size();
const SizeType slave_inactive_size = mSlaveInactiveIndices.size();
const SizeType slave_active_size = mSlaveActiveIndices.size();
const SizeType lm_active_size = mLMActiveIndices.size();
const SizeType total_size = other_dof_size + master_size + slave_inactive_size + slave_active_size;

if (ResidualU.size() != total_size )
ResidualU.resize (total_size, false);

IndexPartition<std::size_t>(other_dof_size).for_each([&](std::size_t i) {
ResidualU[i] = rTotalResidual[mOtherIndices[i]];
});

VectorType aux_res_active_slave(slave_active_size);
IndexPartition<std::size_t>(slave_active_size).for_each([&](std::size_t i) {
aux_res_active_slave[i] = rTotalResidual[mSlaveActiveIndices[i]];
});

if (slave_active_size > 0) {
VectorType aux_complement_master_residual(master_size);
TSparseSpaceType::Mult(mPOperator, aux_res_active_slave, aux_complement_master_residual);

IndexPartition<std::size_t>(master_size).for_each([&](std::size_t i) {
ResidualU[other_dof_size + i] = rTotalResidual[mMasterIndices[i]] - aux_complement_master_residual[i];
});
} else {
IndexPartition<std::size_t>(master_size).for_each([&](std::size_t i) {
ResidualU[other_dof_size + i] = rTotalResidual[mMasterIndices[i]];
});
}

IndexPartition<std::size_t>(slave_inactive_size).for_each([&](std::size_t i) {
ResidualU[other_dof_size + master_size + i] = rTotalResidual[mSlaveInactiveIndices[i]];
});

if (slave_active_size > 0) {
VectorType aux_complement_active_lm_residual(lm_active_size);
TSparseSpaceType::Mult(mCOperator, aux_res_active_slave, aux_complement_active_lm_residual);

IndexPartition<std::size_t>(lm_active_size).for_each([&](std::size_t i) {
ResidualU[other_dof_size + master_size + slave_inactive_size + i] = rTotalResidual[mLMActiveIndices[i]] - aux_complement_active_lm_residual[i];
});
} else {
IndexPartition<std::size_t>(lm_active_size).for_each([&](std::size_t i) {
ResidualU[other_dof_size + master_size + slave_inactive_size + i] = rTotalResidual[mLMActiveIndices[i]];
});
}
}


inline void GetLMAPart(
const VectorType& rTotalResidual,
VectorType& rResidualLMA
)
{
const SizeType other_dof_size = mOtherIndices.size();
const SizeType master_size = mMasterIndices.size();
const SizeType slave_inactive_size = mSlaveInactiveIndices.size();
const SizeType slave_active_size = mSlaveActiveIndices.size();

if (slave_active_size > 0) {

if (rResidualLMA.size() != slave_active_size )
rResidualLMA.resize (slave_active_size, false);

IndexPartition<std::size_t>(rResidualLMA.size()).for_each([&](std::size_t i) {
rResidualLMA[i] = rTotalResidual[mSlaveActiveIndices[i]];
});

VectorType disp_N(other_dof_size);
VectorType disp_M(master_size);
VectorType disp_SI(slave_inactive_size);
VectorType disp_SA(slave_active_size);

IndexPartition<std::size_t>(other_dof_size).for_each([&](std::size_t i) {
disp_N[i] = mDisp[i];
});

IndexPartition<std::size_t>(master_size).for_each([&](std::size_t i) {
disp_M[i] = mDisp[other_dof_size + i];
});

IndexPartition<std::size_t>(slave_inactive_size).for_each([&](std::size_t i) {
disp_SI[i] = mDisp[other_dof_size + master_size + i];
});

IndexPartition<std::size_t>(slave_active_size).for_each([&](std::size_t i) {
disp_SA[i] = mDisp[other_dof_size + master_size + slave_inactive_size + i];
});

VectorType aux_mult(slave_active_size);
TSparseSpaceType::Mult(mKSAN, disp_N, aux_mult);
TSparseSpaceType::UnaliasedAdd (rResidualLMA, -1.0, aux_mult);
TSparseSpaceType::Mult(mKSAM, disp_M, aux_mult);
TSparseSpaceType::UnaliasedAdd (rResidualLMA, -1.0, aux_mult);
if (slave_inactive_size > 0) {
TSparseSpaceType::Mult(mKSASI, disp_SI, aux_mult);
TSparseSpaceType::UnaliasedAdd (rResidualLMA, -1.0, aux_mult);
}
TSparseSpaceType::Mult(mKSASA, disp_SA, aux_mult);
TSparseSpaceType::UnaliasedAdd (rResidualLMA, -1.0, aux_mult);
}
}


inline void GetLMIPart (
const VectorType& rTotalResidual,
VectorType& rResidualLMI
)
{
const SizeType lm_inactive_size = mLMInactiveIndices.size();

if (rResidualLMI.size() != lm_inactive_size )
rResidualLMI.resize (lm_inactive_size, false);

IndexPartition<std::size_t>(lm_inactive_size).for_each([&](std::size_t i) {
rResidualLMI[i] = rTotalResidual[mLMInactiveIndices[i]];
});
}


inline void SetUPart (
VectorType& rTotalResidual,
const VectorType& ResidualU
)
{
const SizeType other_indexes_size = mOtherIndices.size();
const SizeType master_indexes_size = mMasterIndices.size();
const SizeType slave_inactive_indexes_size = mSlaveInactiveIndices.size();
const SizeType slave_active_indexes_size = mSlaveActiveIndices.size();

IndexPartition<std::size_t>(other_indexes_size).for_each([&](std::size_t i) {
rTotalResidual[mOtherIndices[i]] = ResidualU[i];
});
IndexPartition<std::size_t>(master_indexes_size).for_each([&](std::size_t i) {
rTotalResidual[mMasterIndices[i]] = ResidualU[other_indexes_size + i];
});
IndexPartition<std::size_t>(slave_inactive_indexes_size).for_each([&](std::size_t i) {
rTotalResidual[mSlaveInactiveIndices[i]] = ResidualU[other_indexes_size + master_indexes_size + i];
});
IndexPartition<std::size_t>(slave_active_indexes_size).for_each([&](std::size_t i) {
rTotalResidual[mSlaveActiveIndices[i]] = ResidualU[other_indexes_size + master_indexes_size + slave_inactive_indexes_size + i];
});
}


inline void SetLMAPart (
VectorType& rTotalResidual,
const VectorType& ResidualLMA
)
{
IndexPartition<std::size_t>(ResidualLMA.size()).for_each([&](std::size_t i) {
rTotalResidual[mLMActiveIndices[i]] = ResidualLMA[i];
});
}


inline void SetLMIPart (
VectorType& rTotalResidual,
const VectorType& ResidualLMI
)
{
IndexPartition<std::size_t>(ResidualLMI.size()).for_each([&](std::size_t i) {
rTotalResidual[mLMInactiveIndices[i]] = ResidualLMI[i];
});
}


void EnsureStructuralSymmetryMatrix (SparseMatrixType& rA)
{
const SizeType size_system_1 = rA.size1();
const SizeType size_system_2 = rA.size2();
SparseMatrixType transpose(size_system_2, size_system_1);

SparseMatrixMultiplicationUtility::TransposeMatrix<SparseMatrixType, SparseMatrixType>(transpose, rA, 0.0);

SparseMatrixMultiplicationUtility::MatrixAdd<SparseMatrixType, SparseMatrixType>(rA, transpose, 1.0);
}


double CheckMatrix (const SparseMatrixType& rA)
{
const std::size_t* index1 = rA.index1_data().begin();
const std::size_t* index2 = rA.index2_data().begin();
const double* values = rA.value_data().begin();
double norm = 0.0;
for (std::size_t i=0; i<rA.size1(); ++i) {
std::size_t row_begin = index1[i];
std::size_t row_end   = index1[i+1];
if (row_end - row_begin == 0)
KRATOS_WARNING("Checking sparse matrix") << "Line " << i << " has no elements" << std::endl;

for (std::size_t j=row_begin; j<row_end; j++) {
KRATOS_ERROR_IF( index2[j] > rA.size2() ) << "Array above size of A" << std::endl;
norm += values[j]*values[j];
}
}

return std::sqrt (norm);
}


void CreateMatrix(
SparseMatrixType& AuxK,
const SizeType NRows,
const SizeType NCols,
IndexType* Ptr,
IndexType* AuxIndex2,
double* AuxVal
)
{
SparseMatrixMultiplicationUtility::SortRows(Ptr, NRows, NCols, AuxIndex2, AuxVal);

SparseMatrixMultiplicationUtility::CreateSolutionMatrix(AuxK, NRows, NCols, Ptr, AuxIndex2, AuxVal);

delete[] Ptr;
delete[] AuxIndex2;
delete[] AuxVal;
}


void ComputeDiagonalByLumping (
const SparseMatrixType& rA,
SparseMatrixType& rdiagA,
const double Tolerance = ZeroTolerance
)
{
const std::size_t size_A = rA.size1();

IndexType* ptr = new IndexType[size_A + 1];
ptr[0] = 0;
IndexType* aux_index2 = new IndexType[size_A];
double* aux_val = new double[size_A];

IndexPartition<std::size_t>(size_A).for_each([&](std::size_t i) {
ptr[i+1] = i+1;
aux_index2[i] = i;
const double value = rA(i, i);
if (std::abs(value) > Tolerance)
aux_val[i] = 1.0/value;
else 
aux_val[i] = 1.0;
});

SparseMatrixMultiplicationUtility::CreateSolutionMatrix(rdiagA, size_A, size_A, ptr, aux_index2, aux_val);

delete[] ptr;
delete[] aux_index2;
delete[] aux_val;
}


static inline bool IsDisplacementDof(const DofType& rDoF)
{
const auto& r_variable = rDoF.GetVariable();
if (r_variable == DISPLACEMENT_X ||
r_variable == DISPLACEMENT_Y ||
r_variable == DISPLACEMENT_Z) {
return true;
}

return false;
}


static inline bool IsLMDof(const DofType& rDoF)
{
const auto& r_variable = rDoF.GetVariable();
if (r_variable == VECTOR_LAGRANGE_MULTIPLIER_X ||
r_variable == VECTOR_LAGRANGE_MULTIPLIER_Y ||
r_variable == VECTOR_LAGRANGE_MULTIPLIER_Z) {
return true;
}

return false;
}



Parameters GetDefaultParameters()
{
Parameters default_parameters( R"(
{
"solver_type"          : "mixed_ulm_linear_solver",
"tolerance"            : 1.0e-6,
"max_iteration_number" : 200,
"echo_level"           : 0
}  )" );

return default_parameters;
}

}; 

template<class TSparseSpaceType, class TDenseSpaceType, class TPreconditionerType, class TReordererType>
const Kratos::Flags MixedULMLinearSolver<TSparseSpaceType, TDenseSpaceType,TPreconditionerType, TReordererType>::BLOCKS_ARE_ALLOCATED(Kratos::Flags::Create(0));
template<class TSparseSpaceType, class TDenseSpaceType, class TPreconditionerType, class TReordererType>
const Kratos::Flags MixedULMLinearSolver<TSparseSpaceType, TDenseSpaceType,TPreconditionerType, TReordererType>::IS_INITIALIZED(Kratos::Flags::Create(1));

template<class TSparseSpaceType, class TDenseSpaceType, class TPreconditionerType, class TReordererType>
inline std::istream& operator >> (std::istream& IStream,
MixedULMLinearSolver<TSparseSpaceType, TDenseSpaceType,TPreconditionerType, TReordererType>& rThis)
{
return IStream;
}
template<class TSparseSpaceType, class TDenseSpaceType, class TPreconditionerType, class TReordererType>
inline std::ostream& operator << (std::ostream& rOStream,
const MixedULMLinearSolver<TSparseSpaceType, TDenseSpaceType,TPreconditionerType, TReordererType>& rThis)
{
rThis.PrintInfo (rOStream);
rOStream << std::endl;
rThis.PrintData (rOStream);
return rOStream;
}
}  
