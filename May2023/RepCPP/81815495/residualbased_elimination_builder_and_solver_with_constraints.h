#if !defined(KRATOS_RESIDUAL_BASED_ELIMINATION_BUILDER_AND_SOLVER_WITH_CONSTRAINTS)
#define KRATOS_RESIDUAL_BASED_ELIMINATION_BUILDER_AND_SOLVER_WITH_CONSTRAINTS


#include <unordered_set>
#include <unordered_map>




#include "solving_strategies/builder_and_solvers/residualbased_elimination_builder_and_solver.h"
#include "utilities/sparse_matrix_multiplication_utility.h"
#include "utilities/constraint_utilities.h"
#include "input_output/logger.h"
#include "utilities/builtin_timer.h"
#include "utilities/parallel_utilities.h"

namespace Kratos
{







template <class TSparseSpace,
class TDenseSpace,
class TLinearSolver
>
class ResidualBasedEliminationBuilderAndSolverWithConstraints
: public ResidualBasedEliminationBuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver>
{
public:

KRATOS_CLASS_POINTER_DEFINITION(ResidualBasedEliminationBuilderAndSolverWithConstraints);

typedef BuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver> BuilderAndSolverBaseType;

typedef ResidualBasedEliminationBuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver> BaseType;

typedef ResidualBasedEliminationBuilderAndSolverWithConstraints<TSparseSpace, TDenseSpace, TLinearSolver> ClassType;

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
typedef typename BaseType::NodeType NodeType;
typedef typename BaseType::NodesArrayType NodesArrayType;
typedef typename BaseType::ElementsArrayType ElementsArrayType;
typedef typename BaseType::ConditionsArrayType ConditionsArrayType;

typedef PointerVectorSet<Element, IndexedObject> ElementsContainerType;
typedef Element::EquationIdVectorType EquationIdVectorType;
typedef Element::DofsVectorType DofsVectorType;
typedef boost::numeric::ublas::compressed_matrix<double> CompressedMatrixType;

typedef typename NodeType::DofType DofType;
typedef typename DofType::Pointer DofPointerType;

typedef std::unordered_set<IndexType> IndexSetType;

typedef std::unordered_map<IndexType, IndexType> IndexMapType;

typedef MasterSlaveConstraint MasterSlaveConstraintType;
typedef typename MasterSlaveConstraint::Pointer MasterSlaveConstraintPointerType;
typedef std::vector<IndexType> VectorIndexType;
typedef Vector VectorType;




explicit ResidualBasedEliminationBuilderAndSolverWithConstraints() : BaseType()
{
}


explicit ResidualBasedEliminationBuilderAndSolverWithConstraints(
typename TLinearSolver::Pointer pNewLinearSystemSolver,
Parameters ThisParameters
) : BaseType(pNewLinearSystemSolver)
{
ThisParameters = this->ValidateAndAssignParameters(ThisParameters, this->GetDefaultParameters());
this->AssignSettings(ThisParameters);
}


explicit ResidualBasedEliminationBuilderAndSolverWithConstraints(
typename TLinearSolver::Pointer pNewLinearSystemSolver,
const bool CheckConstraintRelation = true,
const bool ResetRelationMatrixEachIteration = false
)
: BaseType(pNewLinearSystemSolver),
mCheckConstraintRelation(CheckConstraintRelation),
mResetRelationMatrixEachIteration(ResetRelationMatrixEachIteration)
{
}


~ResidualBasedEliminationBuilderAndSolverWithConstraints() override
{
}


typename BuilderAndSolverBaseType::Pointer Create(
typename TLinearSolver::Pointer pNewLinearSystemSolver,
Parameters ThisParameters
) const override
{
return Kratos::make_shared<ClassType>(pNewLinearSystemSolver,ThisParameters);
}



void SetUpSystem(ModelPart& rModelPart) override
{
if(rModelPart.MasterSlaveConstraints().size() > 0)
SetUpSystemWithConstraints(rModelPart);
else
BaseType::SetUpSystem(rModelPart);
}


void Build(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemMatrixType& rA,
TSystemVectorType& rb
) override
{
if(rModelPart.MasterSlaveConstraints().size() > 0)
BuildWithConstraints(pScheme, rModelPart, rA, rb);
else
BaseType::Build(pScheme, rModelPart, rA, rb);
}


void BuildAndSolve(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemMatrixType& A,
TSystemVectorType& Dx,
TSystemVectorType& b) override
{
if(rModelPart.MasterSlaveConstraints().size() > 0)
BuildAndSolveWithConstraints(pScheme, rModelPart, A, Dx, b);
else
BaseType::BuildAndSolve(pScheme, rModelPart, A, Dx, b);
}


void BuildRHS(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemVectorType& b) override
{
KRATOS_TRY

if(rModelPart.MasterSlaveConstraints().size() > 0)
BuildRHSWithConstraints(pScheme, rModelPart, b);
else
BaseType::BuildRHS(pScheme, rModelPart, b);

KRATOS_CATCH("")

}


void SetUpDofSet(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart
) override
{
if(rModelPart.MasterSlaveConstraints().size() > 0)
SetUpDofSetWithConstraints(pScheme, rModelPart);
else
BaseType::SetUpDofSet(pScheme, rModelPart);
}


Parameters GetDefaultParameters() const override
{
Parameters default_parameters = Parameters(R"(
{
"name"                                 : "elimination_builder_and_solver_with_constraints",
"check_constraint_relation"            : true,
"reset_relation_matrix_each_iteration" : true
})");

const Parameters base_default_parameters = BaseType::GetDefaultParameters();
default_parameters.RecursivelyAddMissingParameters(base_default_parameters);
return default_parameters;
}


static std::string Name()
{
return "elimination_builder_and_solver_with_constraints";
}




std::string Info() const override
{
return "ResidualBasedEliminationBuilderAndSolverWithConstraints";
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


TSystemMatrixPointerType mpTMatrix = NULL;             
TSystemMatrixPointerType mpOldAMatrix = NULL;          
TSystemVectorPointerType mpConstantVector = NULL;      
TSystemVectorPointerType mpDeltaConstantVector = NULL; 
DofsArrayType mDoFMasterFixedSet;                      
DofsArrayType mDoFSlaveSet;                            
SizeType mDoFToSolveSystemSize = 0;                    
IndexMapType mReactionEquationIdMap;                   

bool mCheckConstraintRelation = false;                 
bool mResetRelationMatrixEachIteration = false;        

bool mComputeConstantContribution = false;             
bool mCleared = true;                                  




void AssembleRelationMatrix(
TSystemMatrixType& rT,
const LocalSystemMatrixType& rTransformationMatrix,
const EquationIdVectorType& rSlaveEquationId,
const EquationIdVectorType& rMasterEquationId
)
{
const SizeType local_size_1 = rTransformationMatrix.size1();

for (IndexType i_local = 0; i_local < local_size_1; ++i_local) {
IndexType i_global = rSlaveEquationId[i_local];

if (i_global < BaseType::mEquationSystemSize) {
BaseType::AssembleRowContributionFreeDofs(rT, rTransformationMatrix, i_global, i_local, rMasterEquationId);
}
}
}


void ConstructMatrixStructure(
typename TSchemeType::Pointer pScheme,
TSystemMatrixType& rA,
ModelPart& rModelPart
) override
{
if(rModelPart.MasterSlaveConstraints().size() > 0)
ConstructMatrixStructureWithConstraints(pScheme, rA, rModelPart);
else
BaseType::ConstructMatrixStructure(pScheme, rA, rModelPart);
}


void BuildAndSolveWithConstraints(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemMatrixType& rA,
TSystemVectorType& rDx,
TSystemVectorType& rb
)
{
KRATOS_TRY

Timer::Start("Build");

ApplyMasterSlaveRelation(pScheme, rModelPart, rA, rDx, rb);

TSystemVectorType dummy_Dx(mDoFToSolveSystemSize);
TSparseSpace::SetToZero(dummy_Dx);
ComputeEffectiveConstant(pScheme, rModelPart, dummy_Dx);

BuildWithConstraints(pScheme, rModelPart, rA, rb);

Timer::Stop("Build");

rDx.resize(mDoFToSolveSystemSize, false);
ApplyDirichletConditions(pScheme, rModelPart, rA, rDx, rb);

KRATOS_INFO_IF("ResidualBasedEliminationBuilderAndSolverWithConstraints", (this->GetEchoLevel() == 3)) <<
"Before the solution of the system" << "\nSystem Matrix = " << rA << "\nUnknowns vector = " << rDx << "\nRHS vector = " << rb << std::endl;

const auto timer = BuiltinTimer();
const double start_solve = timer.ElapsedSeconds();
Timer::Start("Solve");
SystemSolveWithPhysics(rA, rDx, rb, rModelPart);

Timer::Stop("Solve");
const double stop_solve = timer.ElapsedSeconds();

ComputeEffectiveConstant(pScheme, rModelPart, rDx);

const double start_reconstruct_slaves = timer.ElapsedSeconds();
ReconstructSlaveSolutionAfterSolve(pScheme, rModelPart, rA, rDx, rb);

const double stop_reconstruct_slaves = timer.ElapsedSeconds();
KRATOS_INFO_IF("ResidualBasedEliminationBuilderAndSolverWithConstraints", (this->GetEchoLevel() >= 1 && rModelPart.GetCommunicator().MyPID() == 0)) << "Reconstruct slaves time: " << stop_reconstruct_slaves - start_reconstruct_slaves << std::endl;

KRATOS_INFO_IF("ResidualBasedEliminationBuilderAndSolverWithConstraints", (this->GetEchoLevel() >= 1 && rModelPart.GetCommunicator().MyPID() == 0)) << "System solve time: " << stop_solve - start_solve << std::endl;

KRATOS_INFO_IF("ResidualBasedEliminationBuilderAndSolverWithConstraints", (this->GetEchoLevel() == 3)) <<
"After the solution of the system" << "\nSystem Matrix = " << rA << "\nUnknowns vector = " << rDx << "\nRHS vector = " << rb << std::endl;

KRATOS_CATCH("")
}


void BuildRHSWithConstraints(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemVectorType& rb
)
{
Timer::Start("Build RHS");

if(BaseType::mCalculateReactionsFlag) {
TSparseSpace::SetToZero(*(BaseType::mpReactionsVector));
}

BuildRHSNoDirichlet(pScheme,rModelPart,rb);

Timer::Stop("Build RHS");

ApplyDirichletConditionsRHS(pScheme, rModelPart, rb);

const TSystemMatrixType& rTMatrix = *mpTMatrix;

TSystemVectorType rb_copy = rb;
rb.resize(BaseType::mEquationSystemSize, false);
TSparseSpace::Mult(rTMatrix, rb_copy, rb);

TSystemVectorType& r_reactions_vector = *BaseType::mpReactionsVector;

if (BaseType::mCalculateReactionsFlag) {
for (auto& r_dof : BaseType::mDofSet) {
const bool is_master_fixed = mDoFMasterFixedSet.find(r_dof) == mDoFMasterFixedSet.end() ? false : true;
const bool is_slave = mDoFSlaveSet.find(r_dof) == mDoFSlaveSet.end() ? false : true;
if (is_master_fixed || is_slave) { 
const IndexType equation_id = r_dof.EquationId();
r_reactions_vector[mReactionEquationIdMap[equation_id]] += rb[equation_id];
}
}
}

KRATOS_INFO_IF("ResidualBasedEliminationBuilderAndSolverWithConstraints", (this->GetEchoLevel() == 3)) <<
"After the solution of the system" << "\nRHS vector = " << rb << std::endl;
}


void SetUpDofSetWithConstraints(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart
)
{
KRATOS_TRY;

KRATOS_INFO_IF("ResidualBasedEliminationBuilderAndSolverWithConstraints", ( this->GetEchoLevel() > 1 && rModelPart.GetCommunicator().MyPID() == 0)) << "Setting up the dofs" << std::endl;

DofsVectorType dof_list, second_dof_list; 

typedef std::unordered_set < DofPointerType, DofPointerHasher> set_type;

DofsArrayType dof_temp_all, dof_temp_solvable, dof_temp_slave;

BaseType::mDofSet = DofsArrayType(); 
mDoFSlaveSet = DofsArrayType();    


set_type dof_global_set, dof_global_slave_set;

#pragma omp parallel firstprivate(dof_list, second_dof_list)
{
const ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();

set_type dofs_tmp_set, dof_temp_slave_set;
dofs_tmp_set.reserve(20000);
dof_temp_slave_set.reserve(200);

ElementsArrayType& r_elements_array = rModelPart.Elements();
const int number_of_elements = static_cast<int>(r_elements_array.size());
#pragma omp for schedule(guided, 512) nowait
for (int i = 0; i < number_of_elements; ++i) {
auto it_elem = r_elements_array.begin() + i;

pScheme->GetDofList(*it_elem, dof_list, r_current_process_info);
dofs_tmp_set.insert(dof_list.begin(), dof_list.end());
}

ConditionsArrayType& r_conditions_array = rModelPart.Conditions();
const int number_of_conditions = static_cast<int>(r_conditions_array.size());
#pragma omp for  schedule(guided, 512) nowait
for (int i = 0; i < number_of_conditions; ++i) {
auto it_cond = r_conditions_array.begin() + i;

pScheme->GetDofList(*it_cond, dof_list, r_current_process_info);
dofs_tmp_set.insert(dof_list.begin(), dof_list.end());
}

auto& r_constraints_array = rModelPart.MasterSlaveConstraints();
const int number_of_constraints = static_cast<int>(r_constraints_array.size());
#pragma omp for  schedule(guided, 512) nowait
for (int i = 0; i < number_of_constraints; ++i) {
auto it_const = r_constraints_array.begin() + i;

it_const->GetDofList(dof_list, second_dof_list, r_current_process_info);
dofs_tmp_set.insert(dof_list.begin(), dof_list.end());
dofs_tmp_set.insert(second_dof_list.begin(), second_dof_list.end());
dof_temp_slave_set.insert(dof_list.begin(), dof_list.end());
}

#pragma omp critical
{
dof_global_set.insert(dofs_tmp_set.begin(), dofs_tmp_set.end());
dof_global_slave_set.insert(dof_temp_slave_set.begin(), dof_temp_slave_set.end());
}
}

KRATOS_INFO_IF("ResidualBasedEliminationBuilderAndSolverWithConstraints", ( this->GetEchoLevel() > 2)) << "Initializing ordered array filling\n" << std::endl;

dof_temp_all.reserve(dof_global_set.size());
for (auto p_dof : dof_global_set) {
dof_temp_all.push_back( p_dof );
}
dof_temp_all.Sort();
BaseType::mDofSet = dof_temp_all;

dof_temp_slave.reserve(dof_global_slave_set.size());
for (auto p_dof : dof_global_slave_set) {
dof_temp_slave.push_back( p_dof );
}
dof_temp_slave.Sort();
mDoFSlaveSet = dof_temp_slave;

KRATOS_ERROR_IF(BaseType::mDofSet.size() == 0) << "No degrees of freedom!" << std::endl;
KRATOS_WARNING_IF("ResidualBasedEliminationBuilderAndSolverWithConstraints", mDoFSlaveSet.size() == 0) << "No slave degrees of freedom to solve!" << std::endl;

KRATOS_INFO_IF("ResidualBasedEliminationBuilderAndSolverWithConstraints", ( this->GetEchoLevel() > 2)) << "Number of degrees of freedom:" << BaseType::mDofSet.size() << std::endl;

BaseType::mDofSetIsInitialized = true;

KRATOS_INFO_IF("ResidualBasedEliminationBuilderAndSolverWithConstraints", ( this->GetEchoLevel() > 2 && rModelPart.GetCommunicator().MyPID() == 0)) << "Finished setting up the dofs" << std::endl;

#ifdef USE_LOCKS_IN_ASSEMBLY
KRATOS_INFO_IF("ResidualBasedEliminationBuilderAndSolverWithConstraints", ( this->GetEchoLevel() > 2)) << "Initializing lock array" << std::endl;

if (BaseType::mLockArray.size() != 0) {
for (int i = 0; i < static_cast<int>(BaseType::mLockArray.size()); ++i) {
omp_destroy_lock(&BaseType::mLockArray[i]);
}
}
BaseType::mLockArray.resize(BaseType::mDofSet.size());

for (int i = 0; i < static_cast<int>(BaseType::mLockArray.size()); ++i) {
omp_init_lock(&BaseType::mLockArray[i]);
}

KRATOS_INFO_IF("ResidualBasedEliminationBuilderAndSolverWithConstraints", ( this->GetEchoLevel() > 2)) << "End of setup dof set\n" << std::endl;
#endif

#ifdef KRATOS_DEBUG
if(BaseType::GetCalculateReactionsFlag()) {
for(auto dof_iterator = BaseType::mDofSet.begin(); dof_iterator != BaseType::mDofSet.end(); ++dof_iterator) {
KRATOS_ERROR_IF_NOT(dof_iterator->HasReaction()) << "Reaction variable not set for the following : " << std::endl
<< "Node : " << dof_iterator->Id() << std::endl
<< "Dof : " << (*dof_iterator) << std::endl << "Not possible to calculate reactions." << std::endl;
}
}
#endif

KRATOS_CATCH("");
}


void SystemSolveWithPhysics(
TSystemMatrixType& rA,
TSystemVectorType& rDx,
TSystemVectorType& rb,
ModelPart& rModelPart
)
{
KRATOS_TRY

double norm_b = 0.0;
if (TSparseSpace::Size(rb) > 0)
norm_b = TSparseSpace::TwoNorm(rb);

if (norm_b > 0.0) {
DofsArrayType aux_dof_set;
aux_dof_set.reserve(mDoFToSolveSystemSize);
for (auto& r_dof : BaseType::mDofSet) {
if (r_dof.EquationId() < BaseType::mEquationSystemSize) {
auto it = mDoFSlaveSet.find(r_dof);
if (it == mDoFSlaveSet.end())
aux_dof_set.push_back( &r_dof );
}
}
aux_dof_set.Sort();

KRATOS_ERROR_IF_NOT(aux_dof_set.size() == mDoFToSolveSystemSize) << "Inconsistency (I) in system size: " << mDoFToSolveSystemSize << " vs " << aux_dof_set.size() << "\n Size dof set " << BaseType::mDofSet.size() << " vs Size slave dof set " << mDoFSlaveSet.size() << std::endl;
KRATOS_ERROR_IF_NOT(aux_dof_set.size() == rA.size1()) << "Inconsistency (II) in system size: " << rA.size1() << " vs " << aux_dof_set.size() << "\n Size dof set " << BaseType::mDofSet.size() << " vs Size slave dof set " << mDoFSlaveSet.size() << std::endl;

if(BaseType::mpLinearSystemSolver->AdditionalPhysicalDataIsNeeded())
BaseType::mpLinearSystemSolver->ProvideAdditionalData(rA, rDx, rb, aux_dof_set, rModelPart);

BaseType::mpLinearSystemSolver->Solve(rA, rDx, rb);
} else {
TSparseSpace::SetToZero(rDx);
KRATOS_WARNING_IF("ResidualBasedEliminationBuilderAndSolver", rModelPart.GetCommunicator().MyPID() == 0) << "ATTENTION! setting the RHS to zero!" << std::endl;
}

KRATOS_INFO_IF("ResidualBasedEliminationBuilderAndSolver", this->GetEchoLevel() > 1 && rModelPart.GetCommunicator().MyPID() == 0) << *(BaseType::mpLinearSystemSolver) << std::endl;

KRATOS_CATCH("")

}


virtual void ConstructMatrixStructureWithConstraints(
typename TSchemeType::Pointer pScheme,
TSystemMatrixType& rA,
ModelPart& rModelPart
)
{
Timer::Start("MatrixStructure");

const SizeType equation_size = BaseType::mEquationSystemSize;

std::vector<IndexSetType> indices(equation_size);

block_for_each(indices, [](IndexSetType& rIndices){
rIndices.reserve(40);
});

EquationIdVectorType ids(3, 0);
EquationIdVectorType second_ids(3, 0); 

#pragma omp parallel firstprivate(ids, second_ids)
{
const ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();

std::vector<IndexSetType> temp_indexes(equation_size);

#pragma omp for
for (int index = 0; index < static_cast<int>(equation_size); ++index)
temp_indexes[index].reserve(30);

const int number_of_elements = static_cast<int>(rModelPart.Elements().size());

const auto el_begin = rModelPart.ElementsBegin();

#pragma omp for schedule(guided, 512) nowait
for (int i_elem = 0; i_elem<number_of_elements; ++i_elem) {
auto it_elem = el_begin + i_elem;
pScheme->EquationId(*it_elem, ids, r_current_process_info);

for (auto& id_i : ids) {
if (id_i < BaseType::mEquationSystemSize) {
auto& row_indices = temp_indexes[id_i];
for (auto& id_j : ids) {
if (id_j < BaseType::mEquationSystemSize) {
row_indices.insert(id_j);
}
}
}
}
}

const int number_of_conditions = static_cast<int>(rModelPart.Conditions().size());

const auto cond_begin = rModelPart.ConditionsBegin();

#pragma omp for schedule(guided, 512) nowait
for (int i_cond = 0; i_cond<number_of_conditions; ++i_cond) {
auto it_cond = cond_begin + i_cond;
pScheme->EquationId(*it_cond, ids, r_current_process_info);
for (auto& id_i : ids) {
if (id_i < BaseType::mEquationSystemSize) {
auto& row_indices = temp_indexes[id_i];
for (auto& id_j : ids) {
if (id_j < BaseType::mEquationSystemSize) {
row_indices.insert(id_j);
}
}
}
}
}

const int number_of_constraints = static_cast<int>(rModelPart.MasterSlaveConstraints().size());

const auto const_begin = rModelPart.MasterSlaveConstraints().begin();

#pragma omp for schedule(guided, 512) nowait
for (int i_const = 0; i_const < number_of_constraints; ++i_const) {
auto it_const = const_begin + i_const;

if(it_const->IsActive()) {
it_const->EquationIdVector(ids, second_ids, r_current_process_info);
for (auto& id_i : ids) {
if (id_i < BaseType::mEquationSystemSize) {
auto& row_indices = temp_indexes[id_i];
for (auto& id_j : ids) {
if (id_j < BaseType::mEquationSystemSize) {
row_indices.insert(id_j);
}
}
}
}
for (auto& id_i : second_ids) {
if (id_i < BaseType::mEquationSystemSize) {
auto& row_indices = temp_indexes[id_i];
for (auto& id_j : second_ids) {
if (id_j < BaseType::mEquationSystemSize) {
row_indices.insert(id_j);
}
}
}
}
}
}

#pragma omp critical
{
for (int i = 0; i < static_cast<int>(temp_indexes.size()); ++i) {
indices[i].insert(temp_indexes[i].begin(), temp_indexes[i].end());
}
}
}

SizeType nnz = 0;
for (IndexType i = 0; i < indices.size(); ++i)
nnz += indices[i].size();

rA = CompressedMatrixType(indices.size(), indices.size(), nnz);

double *Avalues = rA.value_data().begin();
IndexType *Arow_indices = rA.index1_data().begin();
IndexType *Acol_indices = rA.index2_data().begin();

Arow_indices[0] = 0;
for (int i = 0; i < static_cast<int>(rA.size1()); i++)
Arow_indices[i + 1] = Arow_indices[i] + indices[i].size();

IndexPartition<std::size_t>(rA.size1()).for_each([&](std::size_t Index){
const IndexType row_begin = Arow_indices[Index];
const IndexType row_end = Arow_indices[Index + 1];
IndexType k = row_begin;
for (auto it = indices[Index].begin(); it != indices[Index].end(); ++it) {
Acol_indices[k] = *it;
Avalues[k] = 0.0;
k++;
}

indices[Index].clear(); 

std::sort(&Acol_indices[row_begin], &Acol_indices[row_end]);
});

rA.set_filled(indices.size() + 1, nnz);

Timer::Stop("MatrixStructure");
}


virtual void ConstructRelationMatrixStructure(
typename TSchemeType::Pointer pScheme,
TSystemMatrixType& rT,
ModelPart& rModelPart
)
{
Timer::Start("RelationMatrixStructure");

IndexMapType solvable_dof_reorder;
std::unordered_map<IndexType, IndexSetType> master_indices;

typedef std::pair<IndexType, IndexType> IndexIndexPairType;
typedef std::pair<IndexType, IndexSetType> IndexIndexSetPairType;
IndexType counter = 0;
for (auto& dof : BaseType::mDofSet) {
if (dof.EquationId() < BaseType::mEquationSystemSize) {
const IndexType equation_id = dof.EquationId();
auto it = mDoFSlaveSet.find(dof);
if (it == mDoFSlaveSet.end()) {
solvable_dof_reorder.insert(IndexIndexPairType(equation_id, counter));
master_indices.insert(IndexIndexSetPairType(equation_id, IndexSetType({counter})));
++counter;
} else {
master_indices.insert(IndexIndexSetPairType(equation_id, IndexSetType({})));
}
}
}

const ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();

EquationIdVectorType ids(3, 0);
EquationIdVectorType second_ids(3, 0); 

const int number_of_constraints = static_cast<int>(rModelPart.MasterSlaveConstraints().size());
const auto it_const_begin = rModelPart.MasterSlaveConstraints().begin();
for (int i_const = 0; i_const < number_of_constraints; ++i_const) {
auto it_const = it_const_begin + i_const;

if(it_const->IsActive()) {
it_const->EquationIdVector(ids, second_ids, r_current_process_info);
for (auto& slave_id : ids) {
if (slave_id < BaseType::mEquationSystemSize) {
auto it_slave = solvable_dof_reorder.find(slave_id);
if (it_slave == solvable_dof_reorder.end()) {
for (auto& master_id : second_ids) {
if (master_id < BaseType::mEquationSystemSize) {
auto& master_row_indices = master_indices[slave_id];
master_row_indices.insert(solvable_dof_reorder[master_id]);
}
}
}
}
}
}
}

KRATOS_DEBUG_ERROR_IF_NOT(BaseType::mEquationSystemSize == master_indices.size()) << "Inconsistency in the dofs size: " << BaseType::mEquationSystemSize << "\t vs \t" << master_indices.size() << std::endl;

SizeType nnz = 0;
for (IndexType i = 0; i < BaseType::mEquationSystemSize; ++i) {
nnz += master_indices[i].size();
}

rT = CompressedMatrixType(BaseType::mEquationSystemSize, mDoFToSolveSystemSize, nnz);

double *Tvalues = rT.value_data().begin();
IndexType *Trow_indices = rT.index1_data().begin();
IndexType *Tcol_indices = rT.index2_data().begin();

Trow_indices[0] = 0;
for (IndexType i = 0; i < BaseType::mEquationSystemSize; ++i)
Trow_indices[i + 1] = Trow_indices[i] + master_indices[i].size();

KRATOS_DEBUG_ERROR_IF_NOT(Trow_indices[BaseType::mEquationSystemSize] == nnz) << "Nonzero values does not coincide with the row index definition: " << Trow_indices[BaseType::mEquationSystemSize] << " vs " << nnz << std::endl;

IndexPartition<std::size_t>(rT.size1()).for_each([&](std::size_t Index){
const IndexType row_begin = Trow_indices[Index];
const IndexType row_end = Trow_indices[Index + 1];
IndexType k = row_begin;
for (auto it = master_indices[Index].begin(); it != master_indices[Index].end(); ++it) {
Tcol_indices[k] = *it;
Tvalues[k] = 0.0;
k++;
}

master_indices[Index].clear(); 

std::sort(&Tcol_indices[row_begin], &Tcol_indices[row_end]);
});
rT.set_filled(BaseType::mEquationSystemSize + 1, nnz);

for (auto& solv_dof : solvable_dof_reorder) {
rT(solv_dof.first, solv_dof.second) = 1.0;
}

Timer::Stop("RelationMatrixStructure");
}


void BuildWithConstraints(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemMatrixType& rA,
TSystemVectorType& rb,
const bool UseBaseBuild = true
)
{
KRATOS_TRY

if (UseBaseBuild)
BaseType::Build(pScheme, rModelPart, rA, rb);
else
BuildWithoutConstraints(pScheme, rModelPart, rA, rb);

const auto timer = BuiltinTimer();

const TSystemMatrixType& rTMatrix = *mpTMatrix;

if (mCleared) {
mCleared = false;
ComputeConstraintContribution(pScheme, rModelPart, true, mComputeConstantContribution);
} else if (mResetRelationMatrixEachIteration) {
ResetConstraintSystem();
ComputeConstraintContribution(pScheme, rModelPart, mResetRelationMatrixEachIteration, mComputeConstantContribution);
}

TSystemMatrixType T_transpose_matrix(mDoFToSolveSystemSize, BaseType::mEquationSystemSize);
SparseMatrixMultiplicationUtility::TransposeMatrix<TSystemMatrixType, TSystemMatrixType>(T_transpose_matrix, rTMatrix, 1.0);

TSystemVectorType rb_copy = rb;
if (mComputeConstantContribution) {
TSystemVectorType& rDeltaConstantVector = *mpDeltaConstantVector;
TSystemVectorType aux_constant_vector(rDeltaConstantVector);
TSparseSpace::Mult(rA, rDeltaConstantVector, aux_constant_vector);
TSparseSpace::UnaliasedAdd(rb_copy, -1.0, aux_constant_vector);
}

TSystemMatrixType auxiliar_A_matrix(mDoFToSolveSystemSize, BaseType::mEquationSystemSize);
SparseMatrixMultiplicationUtility::MatrixMultiplication(T_transpose_matrix, rA, auxiliar_A_matrix);

if (mpOldAMatrix == NULL) { 
TSystemMatrixPointerType pNewOldAMatrix = TSystemMatrixPointerType(new TSystemMatrixType(0, 0));
mpOldAMatrix.swap(pNewOldAMatrix);
}
(*mpOldAMatrix).swap(rA);
rA.resize(mDoFToSolveSystemSize, mDoFToSolveSystemSize, false);
rb.resize(mDoFToSolveSystemSize, false);

SparseMatrixMultiplicationUtility::MatrixMultiplication(auxiliar_A_matrix, rTMatrix, rA);
TSparseSpace::Mult(T_transpose_matrix, rb_copy, rb);

auxiliar_A_matrix.resize(0, 0, false);
T_transpose_matrix.resize(0, 0, false);

KRATOS_INFO_IF("ResidualBasedEliminationBuilderAndSolverWithConstraints", this->GetEchoLevel() >= 1) << "Constraint relation build time and multiplication: " << timer.ElapsedSeconds() << std::endl;

KRATOS_INFO_IF("ResidualBasedEliminationBuilderAndSolverWithConstraints", this->GetEchoLevel() > 2) << "Finished parallel building with constraints" << std::endl;

KRATOS_CATCH("")
}


void BuildRHSNoDirichlet(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemVectorType& rb
)
{
KRATOS_TRY

const auto timer = BuiltinTimer();

const TSystemMatrixType& rTMatrix = *mpTMatrix;

if (mCleared) {
mCleared = false;
ComputeConstraintContribution(pScheme, rModelPart, true, mComputeConstantContribution);
} else if (mResetRelationMatrixEachIteration) {
ResetConstraintSystem();
ComputeConstraintContribution(pScheme, rModelPart, mResetRelationMatrixEachIteration, mComputeConstantContribution);
}

TSystemMatrixType T_transpose_matrix(mDoFToSolveSystemSize, BaseType::mEquationSystemSize);
SparseMatrixMultiplicationUtility::TransposeMatrix<TSystemMatrixType, TSystemMatrixType>(T_transpose_matrix, rTMatrix, 1.0);

TSystemMatrixType A; 
if (mComputeConstantContribution) {
A.resize(BaseType::mEquationSystemSize, BaseType::mEquationSystemSize, false);
ConstructMatrixStructure(pScheme, A, rModelPart);
BuildWithoutConstraints(pScheme, rModelPart, A, rb);
} else {
BuildRHSNoDirichletWithoutConstraints(pScheme, rModelPart, rb);
}

TSystemVectorType rb_copy = rb;
if (mComputeConstantContribution) {
TSystemVectorType& rDeltaConstantVector = *mpDeltaConstantVector;
TSystemVectorType aux_constant_vector(rDeltaConstantVector);
TSparseSpace::Mult(A, rDeltaConstantVector, aux_constant_vector);
TSparseSpace::UnaliasedAdd(rb_copy, -1.0, aux_constant_vector);
}

rb.resize(mDoFToSolveSystemSize, false);

TSparseSpace::Mult(T_transpose_matrix, rb_copy, rb);

KRATOS_INFO_IF("ResidualBasedEliminationBuilderAndSolverWithConstraints", this->GetEchoLevel() >= 1) << "Constraint relation build time and multiplication: " << timer.ElapsedSeconds() << std::endl;

KRATOS_INFO_IF("ResidualBasedEliminationBuilderAndSolverWithConstraints", this->GetEchoLevel() > 2) << "Finished parallel building with constraints" << std::endl;

KRATOS_CATCH("")
}


void ResizeAndInitializeVectors(
typename TSchemeType::Pointer pScheme,
TSystemMatrixPointerType& pA,
TSystemVectorPointerType& pDx,
TSystemVectorPointerType& pb,
ModelPart& rModelPart
) override
{
BaseType::ResizeAndInitializeVectors(pScheme, pA, pDx, pb, rModelPart);

if (BaseType::mCalculateReactionsFlag) {
const SizeType reactions_vector_size = BaseType::mDofSet.size() - mDoFToSolveSystemSize + mDoFMasterFixedSet.size();
TSystemVectorType& rReactionsVector = *(BaseType::mpReactionsVector);
if (rReactionsVector.size() != reactions_vector_size)
rReactionsVector.resize(reactions_vector_size, false);
}

if(rModelPart.MasterSlaveConstraints().size() > 0) {
if (mpTMatrix == NULL) { 
TSystemMatrixPointerType pNewT = TSystemMatrixPointerType(new TSystemMatrixType(0, 0));
mpTMatrix.swap(pNewT);
}

if (mpConstantVector == NULL) { 
TSystemVectorPointerType pNewConstantVector = TSystemVectorPointerType(new TSystemVectorType(0));
mpConstantVector.swap(pNewConstantVector);
}

if (mpDeltaConstantVector == NULL) { 
TSystemVectorPointerType pNewConstantVector = TSystemVectorPointerType(new TSystemVectorType(0));
mpDeltaConstantVector.swap(pNewConstantVector);
}

TSystemMatrixType& rTMatrix = *mpTMatrix;
TSystemVectorType& rConstantVector = *mpConstantVector;
TSystemVectorType& rDeltaConstantVector = *mpDeltaConstantVector;

if (rTMatrix.size1() == 0 || BaseType::GetReshapeMatrixFlag() || mCleared) { 
rTMatrix.resize(BaseType::mEquationSystemSize, mDoFToSolveSystemSize, false);
ConstructRelationMatrixStructure(pScheme, rTMatrix, rModelPart);
} else {
if (rTMatrix.size1() != BaseType::mEquationSystemSize || rTMatrix.size2() != mDoFToSolveSystemSize) {
KRATOS_ERROR <<"The equation system size has changed during the simulation. This is not permitted."<<std::endl;
rTMatrix.resize(BaseType::mEquationSystemSize, mDoFToSolveSystemSize, false);
ConstructRelationMatrixStructure(pScheme, rTMatrix, rModelPart);
}
}

if (rConstantVector.size() != BaseType::mEquationSystemSize || BaseType::GetReshapeMatrixFlag() || mCleared) {
rConstantVector.resize(BaseType::mEquationSystemSize, false);
mComputeConstantContribution = ComputeConstraintContribution(pScheme, rModelPart);
} else {
if (rConstantVector.size() != BaseType::mEquationSystemSize) {
KRATOS_ERROR <<"The equation system size has changed during the simulation. This is not permitted."<<std::endl;
rConstantVector.resize(BaseType::mEquationSystemSize, false);
mComputeConstantContribution = ComputeConstraintContribution(pScheme, rModelPart);
}
}
if (mComputeConstantContribution) {
if (rDeltaConstantVector.size() != BaseType::mEquationSystemSize || BaseType::GetReshapeMatrixFlag() || mCleared) {
rDeltaConstantVector.resize(BaseType::mEquationSystemSize, false);
} else {
if (rDeltaConstantVector.size() != BaseType::mEquationSystemSize) {
KRATOS_ERROR <<"The equation system size has changed during the simulation. This is not permitted."<<std::endl;
rDeltaConstantVector.resize(BaseType::mEquationSystemSize, false);
}
}
}
}
}


void CalculateReactions(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemMatrixType& rA,
TSystemVectorType& rDx,
TSystemVectorType& rb
) override
{
KRATOS_TRY

BuildRHS(pScheme, rModelPart, rb);

TSystemVectorType& r_reactions_vector = *BaseType::mpReactionsVector;

for (auto& r_dof : BaseType::mDofSet) {
if ((r_dof.IsFixed()) || mDoFSlaveSet.find(r_dof) != mDoFSlaveSet.end()) {
r_dof.GetSolutionStepReactionValue() = -r_reactions_vector[mReactionEquationIdMap[r_dof.EquationId()]];
}
}

KRATOS_CATCH("ResidualBasedEliminationBuilderAndSolverWithConstraints::CalculateReactions failed ..");
}


void ApplyDirichletConditions(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemMatrixType& rA,
TSystemVectorType& rDx,
TSystemVectorType& rb
) override
{
KRATOS_TRY;

if (mDoFMasterFixedSet.size() > 0) {
std::vector<double> scaling_factors (mDoFToSolveSystemSize, 0.0);

const auto it_dof_begin = BaseType::mDofSet.begin();
IndexType counter = 0;
for (IndexType i = 0; i < BaseType::mDofSet.size(); ++i) {
auto it_dof = it_dof_begin + i;
const IndexType equation_id = it_dof->EquationId();
if (equation_id < BaseType::mEquationSystemSize ) {
auto it_first_check = mDoFSlaveSet.find(*it_dof);
if (it_first_check == mDoFSlaveSet.end()) {
auto it_second_check = mDoFSlaveSet.find(*it_dof);
if (it_second_check == mDoFSlaveSet.end()) {
if(mDoFMasterFixedSet.find(*it_dof) == mDoFMasterFixedSet.end()) {
scaling_factors[counter] = 1.0;
}
}
counter += 1;
}
}
}

double* Avalues = rA.value_data().begin();
IndexType* Arow_indices = rA.index1_data().begin();
IndexType* Acol_indices = rA.index2_data().begin();

const double zero_tolerance = std::numeric_limits<double>::epsilon();

#pragma omp parallel for
for(int k = 0; k < static_cast<int>(mDoFToSolveSystemSize); ++k) {
const IndexType col_begin = Arow_indices[k];
const IndexType col_end = Arow_indices[k+1];
bool empty = true;
for (IndexType j = col_begin; j < col_end; ++j) {
if(std::abs(Avalues[j]) > zero_tolerance) {
empty = false;
break;
}
}

if(empty) {
rA(k,k) = 1.0;
rb[k] = 0.0;
}
}

IndexPartition<std::size_t>(mDoFToSolveSystemSize).for_each([&](std::size_t Index){
const IndexType col_begin = Arow_indices[Index];
const IndexType col_end = Arow_indices[Index+1];
const double k_factor = scaling_factors[Index];
if (k_factor == 0) {
for (IndexType j = col_begin; j < col_end; ++j)
if (Acol_indices[j] != Index )
Avalues[j] = 0.0;

rb[Index] = 0.0;
} else {
for (IndexType j = col_begin; j < col_end; ++j) {
if(scaling_factors[ Acol_indices[j] ] == 0 ) {
Avalues[j] = 0.0;
}
}
}
});
}

KRATOS_CATCH("");
}


void Clear() override
{
BaseType::Clear();

mDoFMasterFixedSet = DofsArrayType();
mDoFSlaveSet = DofsArrayType();

mReactionEquationIdMap.clear();

if (mpTMatrix != nullptr)
TSparseSpace::Clear(mpTMatrix);
if (mpConstantVector != nullptr)
TSparseSpace::Clear(mpConstantVector);
if (mpDeltaConstantVector != nullptr)
TSparseSpace::Clear(mpDeltaConstantVector);

mCleared = true;

KRATOS_INFO_IF("ResidualBasedEliminationBuilderAndSolverWithConstraints", this->GetEchoLevel() > 1) << "Clear Function called" << std::endl;
}


void AssignSettings(const Parameters ThisParameters) override
{
BaseType::AssignSettings(ThisParameters);
mCheckConstraintRelation = ThisParameters["check_constraint_relation"].GetBool();
mResetRelationMatrixEachIteration = ThisParameters["reset_relation_matrix_each_iteration"].GetBool();
}





private:





void SetUpSystemWithConstraints(ModelPart& rModelPart)
{
KRATOS_TRY



const ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();

DofsVectorType slave_dof_list, master_dof_list;

DofsArrayType dof_temp_fixed_master;

typedef std::unordered_set < DofPointerType, DofPointerHasher> set_type;
set_type dof_global_fixed_master_set;

const int number_of_constraints = static_cast<int>(rModelPart.MasterSlaveConstraints().size());
const auto it_const_begin = rModelPart.MasterSlaveConstraints().begin();
#pragma omp parallel firstprivate(slave_dof_list, master_dof_list)
{
set_type dof_temp_fixed_master_set;
dof_temp_fixed_master_set.reserve(2000);

#pragma omp for schedule(guided, 512) nowait
for (int i_const = 0; i_const < number_of_constraints; ++i_const) {
auto it_const = it_const_begin + i_const;

if (it_const->IsActive()) {
it_const->GetDofList(slave_dof_list, master_dof_list, r_current_process_info);

for (auto& master_dof : master_dof_list) {
if (master_dof->IsFixed()) {
dof_temp_fixed_master_set.insert(master_dof);
}
}
}
}

#pragma omp critical
{
dof_global_fixed_master_set.insert(dof_temp_fixed_master_set.begin(), dof_temp_fixed_master_set.end());
}
}

dof_temp_fixed_master.reserve(dof_global_fixed_master_set.size());
for (auto p_dof : dof_global_fixed_master_set) {
dof_temp_fixed_master.push_back( p_dof );
}
dof_temp_fixed_master.Sort();
mDoFMasterFixedSet = dof_temp_fixed_master;

int free_id = 0;
int fix_id = BaseType::mDofSet.size();

for (auto& dof : BaseType::mDofSet) {
if (dof.IsFixed()) {
auto it = mDoFMasterFixedSet.find(dof);
if (it == mDoFMasterFixedSet.end()) {
dof.SetEquationId(--fix_id);
} else {
dof.SetEquationId(free_id++);
}
} else {
dof.SetEquationId(free_id++);
}
}

BaseType::mEquationSystemSize = fix_id;

IndexType counter = 0;
for (auto& dof : BaseType::mDofSet) {
if (dof.EquationId() < BaseType::mEquationSystemSize) {
auto it = mDoFSlaveSet.find(dof);
if (it == mDoFSlaveSet.end()) {
++counter;
}
}
}

mDoFToSolveSystemSize = counter;

counter = 0;
for (auto& r_dof : BaseType::mDofSet) {
const bool is_master_fixed = mDoFMasterFixedSet.find(r_dof) == mDoFMasterFixedSet.end() ? false : true;
const bool is_slave = mDoFSlaveSet.find(r_dof) == mDoFSlaveSet.end() ? false : true;
if (is_master_fixed || is_slave) { 
mReactionEquationIdMap.insert({r_dof.EquationId(), counter});
++counter;
}
}

KRATOS_CATCH("ResidualBasedEliminationBuilderAndSolverWithConstraints::SetUpSystemWithConstraints failed ..");
}


void ApplyMasterSlaveRelation(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemMatrixType& rA,
TSystemVectorType& rDx,
TSystemVectorType& rb
)
{
KRATOS_TRY

ConstraintUtilities::ResetSlaveDofs(rModelPart);

ConstraintUtilities::ApplyConstraints(rModelPart);

KRATOS_CATCH("");
}


bool CheckMasterSlaveRelation(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemVectorType& rDx,
TSystemVectorType& rDxSolved
)
{
KRATOS_TRY

const auto it_dof_begin = BaseType::mDofSet.begin();
TSystemVectorType current_solution(mDoFToSolveSystemSize);
TSystemVectorType updated_solution(BaseType::mEquationSystemSize);
TSystemVectorType residual_solution(BaseType::mEquationSystemSize);

IndexType counter = 0;
for (IndexType i = 0; i < BaseType::mDofSet.size(); ++i) {
auto it_dof = it_dof_begin + i;
const IndexType equation_id = it_dof->EquationId();
if (equation_id < BaseType::mEquationSystemSize ) {
auto it = mDoFSlaveSet.find(*it_dof);
if (it == mDoFSlaveSet.end()) {
current_solution[counter] = it_dof->GetSolutionStepValue() + rDxSolved[counter];
counter += 1;
}
}
}

block_for_each(BaseType::mDofSet, [&, this](Dof<double>& rDof){
const IndexType equation_id = rDof.EquationId();
if (equation_id < this->mEquationSystemSize ) {
residual_solution[equation_id] = rDof.GetSolutionStepValue() + rDx[equation_id];
}
});

const TSystemMatrixType& rTMatrix = *mpTMatrix;
TSparseSpace::Mult(rTMatrix, current_solution, updated_solution);

if (mComputeConstantContribution) {
ComputeConstraintContribution(pScheme, rModelPart, false, true);
const TSystemVectorType& rConstantVector = *mpConstantVector;
TSparseSpace::UnaliasedAdd(updated_solution, 1.0, rConstantVector);
}

TSparseSpace::UnaliasedAdd(residual_solution, -1.0, updated_solution);

for(int k = 0; k < static_cast<int>(BaseType::mEquationSystemSize); ++k) {
if (std::abs(residual_solution[k]) > std::numeric_limits<double>::epsilon()) return false;
}

return true;

KRATOS_CATCH("");
}


void ReconstructSlaveSolutionAfterSolve(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemMatrixType& rA,
TSystemVectorType& rDx,
TSystemVectorType& rb
)
{
KRATOS_TRY

const TSystemMatrixType& rTMatrix = *mpTMatrix;

TSystemVectorType Dx_copy = rDx;
rDx.resize(BaseType::mEquationSystemSize);
TSparseSpace::Mult(rTMatrix, Dx_copy, rDx);

if (mComputeConstantContribution) {
const TSystemVectorType& rDeltaConstantVector = *mpDeltaConstantVector;
TSparseSpace::UnaliasedAdd(rDx, 1.0, rDeltaConstantVector);
}

if (mCheckConstraintRelation) {
KRATOS_ERROR_IF_NOT(CheckMasterSlaveRelation(pScheme, rModelPart, rDx, Dx_copy)) << "The relation between master/slave dofs is not respected" << std::endl;
}

(rA).swap(*mpOldAMatrix);
mpOldAMatrix = NULL;

TSystemVectorType rb_copy = rb;
rb.resize(BaseType::mEquationSystemSize, false);
TSparseSpace::Mult(rTMatrix, rb_copy, rb);

KRATOS_CATCH("ResidualBasedEliminationBuilderAndSolverWithConstraints::ReconstructSlaveSolutionAfterSolve failed ..");
}


void BuildWithoutConstraints(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemMatrixType& rA,
TSystemVectorType& rb
)
{
const ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();

ElementsArrayType& r_elements_array = rModelPart.Elements();

ConditionsArrayType& r_conditons_array = rModelPart.Conditions();

LocalSystemMatrixType lhs_contribution = LocalSystemMatrixType(0, 0);
LocalSystemVectorType rhs_contribution = LocalSystemVectorType(0);

Element::EquationIdVectorType equation_id;

#pragma omp parallel firstprivate( lhs_contribution, rhs_contribution, equation_id)
{
const auto it_elem_begin = r_elements_array.begin();
const int nelements = static_cast<int>(r_elements_array.size());
#pragma omp for schedule(guided, 512) nowait
for (int i = 0; i<nelements; ++i) {
auto it_elem = it_elem_begin + i;
if (it_elem->IsActive()) {
pScheme->CalculateSystemContributions(*it_elem, lhs_contribution, rhs_contribution, equation_id, r_current_process_info);

AssembleWithoutConstraints(rA, rb, lhs_contribution, rhs_contribution, equation_id);
}
}

const auto it_cond_begin = r_conditons_array.begin();
const int nconditions = static_cast<int>(r_conditons_array.size());
#pragma omp for schedule(guided, 512)
for (int i = 0; i<nconditions; ++i) {
auto it_cond = it_cond_begin + i;
if (it_cond->IsActive()) {
pScheme->CalculateSystemContributions(*it_cond, lhs_contribution, rhs_contribution, equation_id, r_current_process_info);

AssembleWithoutConstraints(rA, rb, lhs_contribution, rhs_contribution, equation_id);
}
}
}
}


void BuildRHSNoDirichletWithoutConstraints(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemVectorType& rb
)
{
const ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();

ElementsArrayType& r_elements_array = rModelPart.Elements();

ConditionsArrayType& r_conditons_array = rModelPart.Conditions();

LocalSystemVectorType rhs_contribution = LocalSystemVectorType(0);

Element::EquationIdVectorType equation_id;

#pragma omp parallel firstprivate( rhs_contribution, equation_id)
{
const auto it_elem_begin = r_elements_array.begin();
const int nelements = static_cast<int>(r_elements_array.size());
#pragma omp for schedule(guided, 512) nowait
for (int i = 0; i<nelements; ++i) {
auto it_elem = it_elem_begin + i;
if (it_elem->IsActive()) {
pScheme->CalculateRHSContribution(*it_elem, rhs_contribution, equation_id, r_current_process_info);

AssembleRHSWithoutConstraints(rb, rhs_contribution, equation_id);
}
}

const auto it_cond_begin = r_conditons_array.begin();
const int nconditions = static_cast<int>(r_conditons_array.size());
#pragma omp for schedule(guided, 512)
for (int i = 0; i<nconditions; ++i) {
auto it_cond = it_cond_begin + i;
if (it_cond->IsActive()) {
pScheme->CalculateRHSContribution(*it_cond, rhs_contribution, equation_id, r_current_process_info);

AssembleRHSWithoutConstraints(rb, rhs_contribution, equation_id);
}
}
}
}


void AssembleWithoutConstraints(
TSystemMatrixType& rA,
TSystemVectorType& rb,
const LocalSystemMatrixType& rLHSContribution,
const LocalSystemVectorType& rRHSContribution,
const Element::EquationIdVectorType& rEquationId
)
{
const SizeType local_size = rLHSContribution.size1();

AssembleRHSWithoutConstraints(rb, rRHSContribution, rEquationId);

for (IndexType i_local = 0; i_local < local_size; ++i_local) {
const IndexType i_global = rEquationId[i_local];

if (i_global < BaseType::mEquationSystemSize) {
BaseType::AssembleRowContributionFreeDofs(rA, rLHSContribution, i_global, i_local, rEquationId);
}
}
}



void AssembleRHSWithoutConstraints(
TSystemVectorType& rb,
const LocalSystemVectorType& rRHSContribution,
const Element::EquationIdVectorType& rEquationId
)
{
const SizeType local_size = rRHSContribution.size();

if (!BaseType::mCalculateReactionsFlag) {
for (IndexType i_local = 0; i_local < local_size; ++i_local) {
const IndexType i_global = rEquationId[i_local];

if (i_global < BaseType::mEquationSystemSize) { 
double& r_b_value = rb[i_global];
const double rhs_value = rRHSContribution[i_local];

AtomicAdd(r_b_value, rhs_value);
}
}
} else {
TSystemVectorType& r_reactions_vector = *BaseType::mpReactionsVector;
for (IndexType i_local = 0; i_local < local_size; ++i_local) {
const IndexType i_global = rEquationId[i_local];
auto it_dof = BaseType::mDofSet.begin() + i_global;

const bool is_master_fixed = mDoFMasterFixedSet.find(*it_dof) == mDoFMasterFixedSet.end() ? false : true;
const bool is_slave = mDoFSlaveSet.find(*it_dof) == mDoFSlaveSet.end() ? false : true;
if (is_master_fixed || is_slave) { 
double& r_b_value = r_reactions_vector[mReactionEquationIdMap[i_global]];
const double rhs_value = rRHSContribution[i_local];

AtomicAdd(r_b_value, rhs_value);
} else if (it_dof->IsFree()) {  
double& r_b_value = rb[i_global];
const double& rhs_value = rRHSContribution[i_local];

AtomicAdd(r_b_value, rhs_value);
}
}
}
}


void ResetConstraintSystem()
{
TSystemMatrixType& rTMatrix = *mpTMatrix;
double *Tvalues = rTMatrix.value_data().begin();

IndexPartition<std::size_t>(rTMatrix.nnz()).for_each([&Tvalues](std::size_t Index){
Tvalues[Index] = 0.0;
});

IndexMapType solvable_dof_reorder;

typedef std::pair<IndexType, IndexType> IndexIndexPairType;
IndexType counter = 0;
for (auto& dof : BaseType::mDofSet) {
if (dof.EquationId() < BaseType::mEquationSystemSize) {
const IndexType equation_id = dof.EquationId();
auto it = mDoFSlaveSet.find(dof);
if (it == mDoFSlaveSet.end()) {
solvable_dof_reorder.insert(IndexIndexPairType(equation_id, counter));
++counter;
}
}
}

for (auto& solv_dof : solvable_dof_reorder) {
rTMatrix(solv_dof.first, solv_dof.second) = 1.0;
}

if (mComputeConstantContribution) {
TSystemVectorType& rConstantVector = *mpConstantVector;
TSparseSpace::SetToZero(rConstantVector);
}
}


void ApplyDirichletConditionsRHS(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemVectorType& rb
)
{
KRATOS_TRY;

if (mDoFMasterFixedSet.size() > 0) {
const auto it_dof_begin = BaseType::mDofSet.begin();

IndexPartition<std::size_t>(mDoFToSolveSystemSize).for_each([&, this](std::size_t Index){
auto it_dof = it_dof_begin + Index;
if (Index < this->mEquationSystemSize) {
auto it = mDoFSlaveSet.find(*it_dof);
if (it == mDoFSlaveSet.end()) {
if(mDoFMasterFixedSet.find(*it_dof) != mDoFMasterFixedSet.end()) {
rb[Index] = 0.0;
}
}
}
});
}

KRATOS_CATCH("");
}


bool ComputeConstraintContribution(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
const bool ComputeTranslationMatrix = false,
const bool ComputeConstantVector = false
)
{
KRATOS_TRY;

TSystemMatrixType& rTMatrix = *mpTMatrix;
TSystemVectorType& rConstantVector = *mpConstantVector;

if (ComputeConstantVector) {
IndexPartition<std::size_t>(this->mEquationSystemSize).for_each([&rConstantVector](std::size_t Index){
rConstantVector[Index] = 0.0;
});
}

IndexMapType solvable_dof_reorder;

typedef std::pair<IndexType, IndexType> IndexIndexPairType;
IndexType counter = 0;
for (auto& dof : BaseType::mDofSet) {
if (dof.EquationId() < BaseType::mEquationSystemSize) {
const IndexType equation_id = dof.EquationId();
auto it = mDoFSlaveSet.find(dof);
if (it == mDoFSlaveSet.end()) {
solvable_dof_reorder.insert(IndexIndexPairType(equation_id, counter));
++counter;
}
}
}

const ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();

double aux_constant_value = 0.0;

LocalSystemMatrixType transformation_matrix = LocalSystemMatrixType(0, 0);
LocalSystemVectorType constant_vector = LocalSystemVectorType(0);

EquationIdVectorType slave_equation_id, master_equation_id;

const int number_of_constraints = static_cast<int>(rModelPart.MasterSlaveConstraints().size());

std::unordered_set<IndexType> auxiliar_constant_equations_ids;

#pragma omp parallel firstprivate(transformation_matrix, constant_vector, slave_equation_id, master_equation_id)
{
std::unordered_set<IndexType> auxiliar_temp_constant_equations_ids;
auxiliar_temp_constant_equations_ids.reserve(2000);

#pragma omp for schedule(guided, 512)
for (int i_const = 0; i_const < number_of_constraints; ++i_const) {
auto it_const = rModelPart.MasterSlaveConstraints().begin() + i_const;

if (it_const->IsActive()) {
it_const->CalculateLocalSystem(transformation_matrix, constant_vector, r_current_process_info);
it_const->EquationIdVector(slave_equation_id, master_equation_id, r_current_process_info);

for (auto& id : master_equation_id) {
id = solvable_dof_reorder[id];
}

if (ComputeConstantVector) {
for (IndexType i = 0; i < slave_equation_id.size(); ++i) {
const IndexType i_global = slave_equation_id[i];
if (i_global < BaseType::mEquationSystemSize) {
const double constant_value = constant_vector[i];
if (std::abs(constant_value) > 0.0) {
auxiliar_temp_constant_equations_ids.insert(i_global);
double& r_value = rConstantVector[i_global];
AtomicAdd(r_value, constant_value);
}
}
}
} else {
for (IndexType i = 0; i < slave_equation_id.size(); ++i) {
const IndexType i_global = slave_equation_id[i];
if (i_global < BaseType::mEquationSystemSize) {
const double constant_value = constant_vector[i];
AtomicAdd(aux_constant_value, std::abs(constant_value));
}
}
}

if (ComputeTranslationMatrix) {
AssembleRelationMatrix(rTMatrix, transformation_matrix, slave_equation_id, master_equation_id);
}
}
}

#pragma omp critical
{
auxiliar_constant_equations_ids.insert(auxiliar_temp_constant_equations_ids.begin(), auxiliar_temp_constant_equations_ids.end());
}
}

return aux_constant_value > std::numeric_limits<double>::epsilon() ? true : false;

KRATOS_CATCH("");
}


void ComputeEffectiveConstant(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemVectorType& rDxSolved
)
{
if (mComputeConstantContribution) {
const TSystemMatrixType& rTMatrix = *mpTMatrix;
TSystemVectorType& rConstantVector = *mpConstantVector;
TSystemVectorType& rDeltaConstantVector = *mpDeltaConstantVector;
TSparseSpace::Copy(rConstantVector, rDeltaConstantVector);

TSystemVectorType Dx(BaseType::mEquationSystemSize);
TSparseSpace::Mult(rTMatrix, rDxSolved, Dx);

const auto it_dof_begin = BaseType::mDofSet.begin();

TSystemVectorType u(BaseType::mEquationSystemSize);

block_for_each(BaseType::mDofSet, [&, this](Dof<double>& rDof){
const IndexType equation_id = rDof.EquationId();
if (equation_id < this->mEquationSystemSize ) {
u[equation_id] = rDof.GetSolutionStepValue() + Dx[equation_id];
}
});

TSystemVectorType u_bar(mDoFToSolveSystemSize);
IndexType counter = 0;
for (IndexType i = 0; i < BaseType::mDofSet.size(); ++i) {
auto it_dof = it_dof_begin + i;
const IndexType equation_id = it_dof->EquationId();
if (equation_id < BaseType::mEquationSystemSize ) {

auto it = mDoFSlaveSet.find(*it_dof);
if (it == mDoFSlaveSet.end()) {
u_bar[counter] = it_dof->GetSolutionStepValue() + rDxSolved[counter];
counter += 1;
}
}
}
TSystemVectorType u_bar_complete(BaseType::mEquationSystemSize);
TSparseSpace::Mult(rTMatrix, u_bar, u_bar_complete);

TSparseSpace::UnaliasedAdd(rDeltaConstantVector, 1.0, u_bar_complete);
TSparseSpace::UnaliasedAdd(rDeltaConstantVector, -1.0, u);
}
}




}; 




} 

#endif 
