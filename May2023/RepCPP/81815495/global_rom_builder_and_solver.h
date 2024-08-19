
#pragma once


#include "concurrentqueue/concurrentqueue.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "includes/define.h"
#include "includes/model_part.h"
#include "solving_strategies/schemes/scheme.h"
#include "solving_strategies/builder_and_solvers/builder_and_solver.h"
#include "solving_strategies/builder_and_solvers/residualbased_block_builder_and_solver.h"
#include "utilities/builtin_timer.h"
#include "utilities/reduction_utilities.h"
#include "custom_utilities/ublas_wrapper.h"


#include "rom_application_variables.h"
#include "custom_utilities/rom_auxiliary_utilities.h"

namespace Kratos
{






template <class TSparseSpace, class TDenseSpace, class TLinearSolver>
class GlobalROMBuilderAndSolver : public ResidualBasedBlockBuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver>
{
public:




KRATOS_CLASS_POINTER_DEFINITION(GlobalROMBuilderAndSolver);

using SizeType = std::size_t;
using IndexType = std::size_t;

using ClassType = GlobalROMBuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver>;

using BaseBuilderAndSolverType = BuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver>;
using BaseType = ResidualBasedBlockBuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver>;
using TSchemeType = typename BaseBuilderAndSolverType::TSchemeType;
using DofsArrayType = typename BaseBuilderAndSolverType::DofsArrayType;
using TSystemMatrixType = typename BaseBuilderAndSolverType::TSystemMatrixType;
using TSystemVectorType = typename BaseBuilderAndSolverType::TSystemVectorType;
using LocalSystemVectorType = typename BaseBuilderAndSolverType::LocalSystemVectorType;
using LocalSystemMatrixType = typename BaseBuilderAndSolverType::LocalSystemMatrixType;
using TSystemMatrixPointerType = typename BaseBuilderAndSolverType::TSystemMatrixPointerType;
using TSystemVectorPointerType = typename BaseBuilderAndSolverType::TSystemVectorPointerType;
using ElementsArrayType = typename BaseBuilderAndSolverType::ElementsArrayType;
using ConditionsArrayType = typename BaseBuilderAndSolverType::ConditionsArrayType;

using MasterSlaveConstraintContainerType = typename ModelPart::MasterSlaveConstraintContainerType;
using EquationIdVectorType = typename Element::EquationIdVectorType;
using DofsVectorType = typename Element::DofsVectorType;
using CompressedMatrixType = boost::numeric::ublas::compressed_matrix<double>;
using RomSystemMatrixType = LocalSystemMatrixType;
using RomSystemVectorType = LocalSystemVectorType;
using EigenDynamicMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using EigenDynamicVector = Eigen::Matrix<double, Eigen::Dynamic, 1>;
using EigenSparseMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor, int>;

using DofType = typename Node::DofType;
using DofPointerType = typename DofType::Pointer;
using DofQueue = moodycamel::ConcurrentQueue<DofType::Pointer>;


explicit GlobalROMBuilderAndSolver(
typename TLinearSolver::Pointer pNewLinearSystemSolver,
Parameters ThisParameters): BaseType(pNewLinearSystemSolver)
{
Parameters this_parameters_copy = ThisParameters.Clone();
this_parameters_copy = this->ValidateAndAssignParameters(this_parameters_copy, this->GetDefaultParameters());
this->AssignSettings(this_parameters_copy);
}

explicit GlobalROMBuilderAndSolver(
typename TLinearSolver::Pointer pNewLinearSystemSolver)
: ResidualBasedBlockBuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver>(pNewLinearSystemSolver)
{
}

~GlobalROMBuilderAndSolver() = default;



typename BaseBuilderAndSolverType::Pointer Create(
typename TLinearSolver::Pointer pNewLinearSystemSolver,
Parameters ThisParameters) const override
{
return Kratos::make_shared<ClassType>(pNewLinearSystemSolver,ThisParameters);
}

void SetUpDofSet(
typename TSchemeType::Pointer pScheme,
ModelPart &rModelPart) override
{
KRATOS_TRY;

KRATOS_INFO_IF("GlobalROMBuilderAndSolver", (this->GetEchoLevel() > 1)) << "Setting up the dofs" << std::endl;
KRATOS_INFO_IF("GlobalROMBuilderAndSolver", (this->GetEchoLevel() > 2)) << "Number of threads" << ParallelUtilities::GetNumThreads() << "\n" << std::endl;
KRATOS_INFO_IF("GlobalROMBuilderAndSolver", (this->GetEchoLevel() > 2)) << "Initializing element loop" << std::endl;

if (mHromWeightsInitialized == false) {
InitializeHROMWeights(rModelPart);
}

auto dof_queue = ExtractDofSet(pScheme, rModelPart);

KRATOS_INFO_IF("GlobalROMBuilderAndSolver", (this->GetEchoLevel() > 2)) << "Initializing ordered array filling\n" << std::endl;
auto dof_array = SortAndRemoveDuplicateDofs(dof_queue);

BaseBuilderAndSolverType::GetDofSet().swap(dof_array);
BaseBuilderAndSolverType::SetDofSetIsInitializedFlag(true);

KRATOS_ERROR_IF(BaseBuilderAndSolverType::GetDofSet().size() == 0) << "No degrees of freedom!" << std::endl;
KRATOS_INFO_IF("GlobalROMBuilderAndSolver", (this->GetEchoLevel() > 2)) << "Number of degrees of freedom:" << BaseBuilderAndSolverType::GetDofSet().size() << std::endl;
KRATOS_INFO_IF("GlobalROMBuilderAndSolver", (this->GetEchoLevel() > 2)) << "Finished setting up the dofs" << std::endl;

#ifdef KRATOS_DEBUG
if (BaseBuilderAndSolverType::GetCalculateReactionsFlag())
{
for (const auto& r_dof: BaseBuilderAndSolverType::GetDofSet())
{
KRATOS_ERROR_IF_NOT(r_dof.HasReaction())
<< "Reaction variable not set for the following :\n"
<< "Node : " << r_dof.Id() << '\n'
<< "Dof  : " << r_dof      << '\n'
<< "Not possible to calculate reactions." << std::endl;
}
}
#endif
KRATOS_CATCH("");
}

void SetUpSystem(ModelPart &rModelPart) override
{
auto& r_dof_set = BaseBuilderAndSolverType::GetDofSet();
BaseBuilderAndSolverType::mEquationSystemSize = r_dof_set.size();
IndexPartition<IndexType>(r_dof_set.size()).for_each([&](IndexType Index)
{
auto dof_iterator = r_dof_set.begin() + Index;
dof_iterator->SetEquationId(Index);
});
}


SizeType GetNumberOfROMModes() const noexcept
{
return mNumberOfRomModes;
} 

void ProjectToFineBasis(
const TSystemVectorType& rRomUnkowns,
const ModelPart& rModelPart,
TSystemVectorType& rDx) const
{
const auto& r_dof_set = BaseBuilderAndSolverType::GetDofSet();
block_for_each(r_dof_set, [&](const DofType& r_dof)
{
const auto& r_node = rModelPart.GetNode(r_dof.Id());
const Matrix& r_rom_nodal_basis = r_node.GetValue(ROM_BASIS);
const Matrix::size_type row_id = mMapPhi.at(r_dof.GetVariable().Key());
rDx[r_dof.EquationId()] = inner_prod(row(r_rom_nodal_basis, row_id), rRomUnkowns);
});
}

void BuildRightROMBasis(
const ModelPart& rModelPart) 
{
const auto& r_dof_set = BaseBuilderAndSolverType::GetDofSet();
block_for_each(r_dof_set, [&](const DofType& r_dof)
{
const auto& r_node = rModelPart.GetNode(r_dof.Id());
const Matrix& r_rom_nodal_basis = r_node.GetValue(ROM_BASIS);
const Matrix::size_type row_id = mMapPhi.at(r_dof.GetVariable().Key());
if (r_dof.IsFixed())
{
noalias(row(mPhiGlobal, r_dof.EquationId())) = ZeroVector(r_rom_nodal_basis.size2());
}
else{
noalias(row(mPhiGlobal, r_dof.EquationId())) = row(r_rom_nodal_basis, row_id);
}
});
}

virtual void InitializeSolutionStep(
ModelPart& rModelPart,
TSystemMatrixType& rA,
TSystemVectorType& rDx,
TSystemVectorType& rb) override
{
BaseBuilderAndSolverType::InitializeSolutionStep(rModelPart, rA, rDx, rb);

auto& r_root_mp = rModelPart.GetRootModelPart();
r_root_mp.GetValue(ROM_SOLUTION_INCREMENT) = ZeroVector(GetNumberOfROMModes());
}


void BuildAndSolve(
typename TSchemeType::Pointer pScheme,
ModelPart &rModelPart,
TSystemMatrixType &A,
TSystemVectorType &Dx,
TSystemVectorType &b) override
{
KRATOS_TRY

BuildAndProjectROM(pScheme, rModelPart, A, b);

SolveROM(rModelPart, A, b, Dx);

KRATOS_CATCH("")
}

void ResizeAndInitializeVectors(
typename TSchemeType::Pointer pScheme,
TSystemMatrixPointerType &pA,
TSystemVectorPointerType &pDx,
TSystemVectorPointerType &pb,
ModelPart &rModelPart) override
{
KRATOS_TRY

if (!pA) {
TSystemMatrixPointerType p_new_A = Kratos::make_shared<TSystemMatrixType>(0, 0);
pA.swap(p_new_A);
}
if (!pDx) {
TSystemVectorPointerType p_new_Dx = Kratos::make_shared<TSystemVectorType>(0);
pDx.swap(p_new_Dx);
}
if (!pb) {
TSystemVectorPointerType p_new_b = Kratos::make_shared<TSystemVectorType>(0);
pb.swap(p_new_b);
}

TSystemVectorType& r_Dx = *pDx;
if (r_Dx.size() != BaseBuilderAndSolverType::GetEquationSystemSize()) {
r_Dx.resize(BaseBuilderAndSolverType::GetEquationSystemSize(), false);
}

TSystemVectorType& r_b = *pb;
if (r_b.size() != BaseBuilderAndSolverType::GetEquationSystemSize()) {
r_b.resize(BaseBuilderAndSolverType::GetEquationSystemSize(), false);
}

KRATOS_CATCH("")
}

Parameters GetDefaultParameters() const override
{
Parameters default_parameters = Parameters(R"(
{
"name" : "global_rom_builder_and_solver",
"nodal_unknowns" : [],
"number_of_rom_dofs" : 10
})");
default_parameters.AddMissingParameters(BaseBuilderAndSolverType::GetDefaultParameters());

return default_parameters;
}

static std::string Name()
{
return "global_rom_builder_and_solver";
}




virtual std::string Info() const override
{
return "GlobalROMBuilderAndSolver";
}

virtual void PrintInfo(std::ostream &rOStream) const override
{
rOStream << Info();
}

virtual void PrintData(std::ostream &rOStream) const override
{
rOStream << Info();
}


protected:


SizeType mNodalDofs;
std::unordered_map<Kratos::VariableData::KeyType, Matrix::size_type> mMapPhi;

ElementsArrayType mSelectedElements;
ConditionsArrayType mSelectedConditions;

bool mHromSimulation = false;
bool mHromWeightsInitialized = false;

bool mRightRomBasisInitialized = false;



void AssignSettings(const Parameters ThisParameters) override
{
BaseBuilderAndSolverType::AssignSettings(ThisParameters);

mNodalDofs = ThisParameters["nodal_unknowns"].size();
mNumberOfRomModes = ThisParameters["number_of_rom_dofs"].GetInt();

IndexType k = 0;
for (const auto& r_var_name : ThisParameters["nodal_unknowns"].GetStringArray()) {
if(KratosComponents<Variable<double>>::Has(r_var_name)) {
const auto& var = KratosComponents<Variable<double>>::Get(r_var_name);
mMapPhi[var.Key()] = k++;
} else {
KRATOS_ERROR << "Variable \""<< r_var_name << "\" not valid" << std::endl;
}
}
}


void InitializeHROMWeights(ModelPart& rModelPart)
{
KRATOS_TRY

using ElementQueue = moodycamel::ConcurrentQueue<Element::Pointer>;
using ConditionQueue = moodycamel::ConcurrentQueue<Condition::Pointer>;

ElementQueue element_queue;
block_for_each(rModelPart.Elements().GetContainer(),
[&](Element::Pointer p_element)
{
if (p_element->Has(HROM_WEIGHT)) {
element_queue.enqueue(std::move(p_element));
} else {
p_element->SetValue(HROM_WEIGHT, 1.0);
}
});

ConditionQueue condition_queue;
block_for_each(rModelPart.Conditions().GetContainer(),
[&](Condition::Pointer p_condition)
{
if (p_condition->Has(HROM_WEIGHT)) {
condition_queue.enqueue(std::move(p_condition));
} else {
p_condition->SetValue(HROM_WEIGHT, 1.0);
}
});

std::size_t err_id;
mSelectedElements.reserve(element_queue.size_approx());
Element::Pointer p_element;
while ( (err_id = element_queue.try_dequeue(p_element)) != 0) {
mSelectedElements.push_back(std::move(p_element));
}

mSelectedConditions.reserve(condition_queue.size_approx());
Condition::Pointer p_condition;
while ( (err_id = condition_queue.try_dequeue(p_condition)) != 0) {
mSelectedConditions.push_back(std::move(p_condition));
}

mHromSimulation = !(mSelectedElements.empty() && mSelectedConditions.empty());
mHromWeightsInitialized = true;

KRATOS_CATCH("")
}

static DofQueue ExtractDofSet(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart)
{
KRATOS_TRY

DofQueue dof_queue;

const auto enqueue_bulk_move = [](DofQueue& r_queue, DofsVectorType& r_dof_list) {
for(auto& p_dof: r_dof_list) {
r_queue.enqueue(std::move(p_dof));
}
r_dof_list.clear();
};

DofsVectorType tls_dof_list; 
block_for_each(rModelPart.Elements(), tls_dof_list,
[&](const Element& r_element, DofsVectorType& r_dof_list)
{
pScheme->GetDofList(r_element, r_dof_list, rModelPart.GetProcessInfo());
enqueue_bulk_move(dof_queue, r_dof_list);
});

block_for_each(rModelPart.Conditions(), tls_dof_list,
[&](const Condition& r_condition, DofsVectorType& r_dof_list)
{
pScheme->GetDofList(r_condition, r_dof_list, rModelPart.GetProcessInfo());
enqueue_bulk_move(dof_queue, r_dof_list);
});

std::pair<DofsVectorType, DofsVectorType> tls_ms_dof_lists; 
block_for_each(rModelPart.MasterSlaveConstraints(), tls_ms_dof_lists,
[&](const MasterSlaveConstraint& r_constraint, std::pair<DofsVectorType, DofsVectorType>& r_dof_lists)
{
r_constraint.GetDofList(r_dof_lists.first, r_dof_lists.second, rModelPart.GetProcessInfo());

enqueue_bulk_move(dof_queue, r_dof_lists.first);
enqueue_bulk_move(dof_queue, r_dof_lists.second);
});

return dof_queue;

KRATOS_CATCH("")
}

static DofsArrayType SortAndRemoveDuplicateDofs(DofQueue& rDofQueue)
{
KRATOS_TRY

DofsArrayType dof_array;
dof_array.reserve(rDofQueue.size_approx());
DofType::Pointer p_dof;
std::size_t err_id;
while ( (err_id = rDofQueue.try_dequeue(p_dof)) != 0) {
dof_array.push_back(std::move(p_dof));
}

dof_array.Unique(); 

return dof_array;

KRATOS_CATCH("")
}


template<typename TMatrix>
static void ResizeIfNeeded(TMatrix& mat, const SizeType rows, const SizeType cols)
{
if(mat.size1() != rows || mat.size2() != cols) {
mat.resize(rows, cols, false);
}
}


virtual void BuildAndProjectROM(
typename TSchemeType::Pointer pScheme,
ModelPart &rModelPart,
TSystemMatrixType &rA,
TSystemVectorType &rb)
{
KRATOS_TRY

KRATOS_ERROR_IF(!pScheme) << "No scheme provided!" << std::endl;

const auto assembling_timer = BuiltinTimer();

if (rA.size1() != BaseType::mEquationSystemSize || rA.size2() != BaseType::mEquationSystemSize) {
rA.resize(BaseType::mEquationSystemSize, BaseType::mEquationSystemSize, false);
BaseType::ConstructMatrixStructure(pScheme, rA, rModelPart);
}

Build(pScheme, rModelPart, rA, rb);

ProjectROM(rModelPart, rA, rb);

double time = assembling_timer.ElapsedSeconds();
KRATOS_INFO_IF("GlobalROMBuilderAndSolver", (this->GetEchoLevel() > 0)) << "Build and project time: " << time << std::endl;

KRATOS_CATCH("")
}


void Build(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemMatrixType& A,
TSystemVectorType& b) override
{
KRATOS_TRY

KRATOS_ERROR_IF(!pScheme) << "No scheme provided!" << std::endl;

const int nelements = mHromSimulation ? mSelectedElements.size() : rModelPart.Elements().size();

const int nconditions = mHromSimulation ? mSelectedConditions.size() : rModelPart.Conditions().size();

const ProcessInfo& CurrentProcessInfo = rModelPart.GetProcessInfo();
ModelPart::ElementsContainerType::iterator el_begin = mHromSimulation ? mSelectedElements.begin() : rModelPart.Elements().begin();
ModelPart::ConditionsContainerType::iterator cond_begin = mHromSimulation ? mSelectedConditions.begin() : rModelPart.Conditions().begin();

LocalSystemMatrixType LHS_Contribution = LocalSystemMatrixType(0, 0);
LocalSystemVectorType RHS_Contribution = LocalSystemVectorType(0);

Element::EquationIdVectorType EquationId;

const auto timer = BuiltinTimer();

#pragma omp parallel firstprivate(nelements,nconditions, LHS_Contribution, RHS_Contribution, EquationId )
{
# pragma omp for  schedule(guided, 512) nowait
for (int k = 0; k < nelements; k++) {
auto it_elem = el_begin + k;

if (it_elem->IsActive()) {
pScheme->CalculateSystemContributions(*it_elem, LHS_Contribution, RHS_Contribution, EquationId, CurrentProcessInfo);

const double h_rom_weight = mHromSimulation ? it_elem->GetValue(HROM_WEIGHT) : 1.0;
LHS_Contribution *= h_rom_weight;
RHS_Contribution *= h_rom_weight;

BaseType::Assemble(A, b, LHS_Contribution, RHS_Contribution, EquationId);
}

}

#pragma omp for  schedule(guided, 512)
for (int k = 0; k < nconditions; k++) {
auto it_cond = cond_begin + k;

if (it_cond->IsActive()) {
pScheme->CalculateSystemContributions(*it_cond, LHS_Contribution, RHS_Contribution, EquationId, CurrentProcessInfo);

const double h_rom_weight = mHromSimulation ? it_cond->GetValue(HROM_WEIGHT) : 1.0;
LHS_Contribution *= h_rom_weight;
RHS_Contribution *= h_rom_weight;

BaseType::Assemble(A, b, LHS_Contribution, RHS_Contribution, EquationId);
}
}
}

KRATOS_INFO_IF("GlobalROMResidualBasedBlockBuilderAndSolver", this->GetEchoLevel() >= 1) << "Build time: " << timer.ElapsedSeconds() << std::endl;

KRATOS_INFO_IF("GlobalROMResidualBasedBlockBuilderAndSolver", (this->GetEchoLevel() > 2 && rModelPart.GetCommunicator().MyPID() == 0)) << "Finished parallel building" << std::endl;

KRATOS_CATCH("")
}


virtual void ProjectROM(
ModelPart &rModelPart,
TSystemMatrixType &rA,
TSystemVectorType &rb)
{
KRATOS_TRY

if (mRightRomBasisInitialized==false){
mPhiGlobal = ZeroMatrix(BaseBuilderAndSolverType::GetEquationSystemSize(), GetNumberOfROMModes());
mRightRomBasisInitialized = true;
}

BuildRightROMBasis(rModelPart);

auto a_wrapper = UblasWrapper<double>(rA);
const auto& eigen_rA = a_wrapper.matrix();
Eigen::Map<EigenDynamicVector> eigen_rb(rb.data().begin(), rb.size());
Eigen::Map<EigenDynamicMatrix> eigen_mPhiGlobal(mPhiGlobal.data().begin(), mPhiGlobal.size1(), mPhiGlobal.size2());

EigenDynamicMatrix eigen_rA_times_mPhiGlobal = eigen_rA * eigen_mPhiGlobal; 

mEigenRomA = eigen_mPhiGlobal.transpose() * eigen_rA_times_mPhiGlobal; 
mEigenRomB = eigen_mPhiGlobal.transpose() * eigen_rb; 

KRATOS_CATCH("")
}


virtual void SolveROM(
ModelPart &rModelPart,
TSystemMatrixType &rA,
TSystemVectorType &rb,
TSystemVectorType &rDx)
{
KRATOS_TRY

RomSystemVectorType dxrom(GetNumberOfROMModes());

const auto solving_timer = BuiltinTimer();

using EigenDynamicVector = Eigen::Matrix<double, Eigen::Dynamic, 1>;
Eigen::Map<EigenDynamicVector> dxrom_eigen(dxrom.data().begin(), dxrom.size());
dxrom_eigen = mEigenRomA.colPivHouseholderQr().solve(mEigenRomB);

double time = solving_timer.ElapsedSeconds();
KRATOS_INFO_IF("GlobalROMBuilderAndSolver", (this->GetEchoLevel() > 0)) << "Solve reduced system time: " << time << std::endl;

auto& r_root_mp = rModelPart.GetRootModelPart();
noalias(r_root_mp.GetValue(ROM_SOLUTION_INCREMENT)) += dxrom;

const auto backward_projection_timer = BuiltinTimer();
ProjectToFineBasis(dxrom, rModelPart, rDx);
KRATOS_INFO_IF("GlobalROMBuilderAndSolver", (this->GetEchoLevel() > 0)) << "Project to fine basis time: " << backward_projection_timer.ElapsedSeconds() << std::endl;

KRATOS_CATCH("")
}



private:

SizeType mNumberOfRomModes;
EigenDynamicMatrix mEigenRomA;
EigenDynamicVector mEigenRomB;
Matrix mPhiGlobal;


}; 




} 