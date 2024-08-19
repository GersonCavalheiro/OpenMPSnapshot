
#pragma once

#include <unordered_set>
#include <unordered_map>


#include "solving_strategies/builder_and_solvers/residualbased_elimination_builder_and_solver_with_constraints.h"

namespace Kratos
{







template<class TSparseSpace,
class TDenseSpace, 
class TLinearSolver 
>
class ContactResidualBasedEliminationBuilderAndSolverWithConstraints
: public ResidualBasedEliminationBuilderAndSolverWithConstraints< TSparseSpace, TDenseSpace, TLinearSolver >
{
public:

KRATOS_CLASS_POINTER_DEFINITION(ContactResidualBasedEliminationBuilderAndSolverWithConstraints);

typedef BuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver> BaseBuilderAndSolverType;

typedef ResidualBasedEliminationBuilderAndSolverWithConstraints< TSparseSpace, TDenseSpace, TLinearSolver > BaseType;

typedef ContactResidualBasedEliminationBuilderAndSolverWithConstraints<TSparseSpace, TDenseSpace, TLinearSolver> ClassType;

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

typedef ModelPart::MasterSlaveConstraintContainerType ConstraintContainerType;

typedef typename BaseType::ElementsContainerType ElementsContainerType;
typedef typename BaseType::EquationIdVectorType EquationIdVectorType;
typedef typename BaseType::DofsVectorType DofsVectorType;

typedef typename BaseType::DofType DofType;
typedef typename BaseType::DofPointerType DofPointerType;

typedef std::vector<typename DofType::Pointer> DofPointerVectorType;

typedef std::size_t SizeType;

typedef std::size_t IndexType;

typedef std::unordered_set<IndexType> IndexSetType;




explicit ContactResidualBasedEliminationBuilderAndSolverWithConstraints() : BaseType()
{
}


explicit ContactResidualBasedEliminationBuilderAndSolverWithConstraints(
typename TLinearSolver::Pointer pNewLinearSystemSolver,
Parameters ThisParameters
) : BaseType(pNewLinearSystemSolver)
{
ThisParameters = this->ValidateAndAssignParameters(ThisParameters, this->GetDefaultParameters());
this->AssignSettings(ThisParameters);
}


ContactResidualBasedEliminationBuilderAndSolverWithConstraints(
typename TLinearSolver::Pointer pNewLinearSystemSolver)
: BaseType(pNewLinearSystemSolver)
{
}


~ContactResidualBasedEliminationBuilderAndSolverWithConstraints() override
{
}




typename BaseBuilderAndSolverType::Pointer Create(
typename TLinearSolver::Pointer pNewLinearSystemSolver,
Parameters ThisParameters
) const override
{
return Kratos::make_shared<ClassType>(pNewLinearSystemSolver,ThisParameters);
}


void SetUpSystem(
ModelPart& rModelPart
) override
{
if(rModelPart.MasterSlaveConstraints().size() > 0)
SetUpSystemWithConstraints(rModelPart);
else
BaseSetUpSystem(rModelPart);
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
"name" : "contact_residual_elimination_builder_and_solver_with_constraints"
})");

const Parameters base_default_parameters = BaseType::GetDefaultParameters();
default_parameters.RecursivelyAddMissingParameters(base_default_parameters);
return default_parameters;
}


static std::string Name()
{
return "contact_residual_elimination_builder_and_solver_with_constraints";
}





protected:





void SetUpDofSetWithConstraints(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart
)
{
KRATOS_TRY;

if (rModelPart.NodesBegin()->SolutionStepsDataHas(VECTOR_LAGRANGE_MULTIPLIER)) {
IndexType constraint_id = 1;
for (auto& constrain : rModelPart.MasterSlaveConstraints()) {
constrain.SetId(constraint_id);
++constraint_id;
}

DofsVectorType dof_list, second_dof_list; 

LocalSystemMatrixType transformation_matrix = LocalSystemMatrixType(0, 0);
LocalSystemVectorType constant_vector = LocalSystemVectorType(0);

const auto& r_clone_constraint = KratosComponents<MasterSlaveConstraint>::Get("LinearMasterSlaveConstraint");

#pragma omp parallel firstprivate(transformation_matrix, constant_vector, dof_list, second_dof_list)
{
ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();

ConstraintContainerType constraints_buffer;

auto& r_constraints_array = rModelPart.MasterSlaveConstraints();
const int number_of_constraints = static_cast<int>(r_constraints_array.size());
#pragma omp for schedule(guided, 512)
for (int i = 0; i < number_of_constraints; ++i) {
auto it_const = r_constraints_array.begin() + i;

it_const->GetDofList(dof_list, second_dof_list, r_current_process_info);
it_const->CalculateLocalSystem(transformation_matrix, constant_vector, r_current_process_info);

DofPointerVectorType slave_dofs, master_dofs;
bool create_lm_constraint = false;

bool slave_nodes_master_dof = false;
for (auto& p_dof : second_dof_list) {
if (IsDisplacementDof(*p_dof)) {
const IndexType node_id = p_dof->Id();
auto pnode = rModelPart.pGetNode(node_id);
if (pnode->Is(SLAVE)) { 
slave_nodes_master_dof = true;
break;
}
}
}

for (auto& p_dof : dof_list) {
if (IsDisplacementDof(*p_dof)) {
const IndexType node_id = p_dof->Id();
const auto& r_variable = p_dof->GetVariable();
auto pnode = rModelPart.pGetNode(node_id);
if (pnode->IsNot(INTERFACE) || slave_nodes_master_dof) { 
if (r_variable == DISPLACEMENT_X) {
slave_dofs.push_back(pnode->pGetDof(VECTOR_LAGRANGE_MULTIPLIER_X));
} else if (r_variable == DISPLACEMENT_Y) {
slave_dofs.push_back(pnode->pGetDof(VECTOR_LAGRANGE_MULTIPLIER_Y));
} else if (r_variable == DISPLACEMENT_Z) {
slave_dofs.push_back(pnode->pGetDof(VECTOR_LAGRANGE_MULTIPLIER_Z));
}
} else { 
it_const->Set(TO_ERASE);
}
}
}
if (slave_nodes_master_dof) { 
for (auto& p_dof : second_dof_list) {
if (IsDisplacementDof(*p_dof)) {
const IndexType node_id = p_dof->Id();
const auto& r_variable = p_dof->GetVariable();
auto pnode = rModelPart.pGetNode(node_id);
if (r_variable == DISPLACEMENT_X) {
master_dofs.push_back(pnode->pGetDof(VECTOR_LAGRANGE_MULTIPLIER_X));
} else if (r_variable == DISPLACEMENT_Y) {
master_dofs.push_back(pnode->pGetDof(VECTOR_LAGRANGE_MULTIPLIER_Y));
} else if (r_variable == DISPLACEMENT_Z) {
master_dofs.push_back(pnode->pGetDof(VECTOR_LAGRANGE_MULTIPLIER_Z));
}
}
}
}

if ((slave_dofs.size() == dof_list.size()) &&
(master_dofs.size() == second_dof_list.size())) {
create_lm_constraint = true;
}

if (create_lm_constraint) {
auto p_constraint = r_clone_constraint.Create(constraint_id + i + 1, master_dofs, slave_dofs, transformation_matrix, constant_vector);
(constraints_buffer).insert((constraints_buffer).begin(), p_constraint);
}
}

#pragma omp critical
{
rModelPart.AddMasterSlaveConstraints(constraints_buffer.begin(),constraints_buffer.end());
}
}
}

rModelPart.RemoveMasterSlaveConstraintsFromAllLevels(TO_ERASE);

KRATOS_INFO_IF("ContactResidualBasedEliminationBuilderAndSolverWithConstraints", (this->GetEchoLevel() > 0)) <<
"Model part after creating new constraints" << rModelPart << std::endl;

BaseType::SetUpDofSetWithConstraints(pScheme, rModelPart);

KRATOS_CATCH("");
}


void AssignSettings(const Parameters ThisParameters) override
{
BaseType::AssignSettings(ThisParameters);
}





private:





void SetUpSystemWithConstraints(ModelPart& rModelPart)
{
KRATOS_TRY

BaseSetUpSystem(rModelPart);

IndexType counter = 0;
for (auto& dof : BaseType::mDofSet) {
if (dof.EquationId() < BaseType::mEquationSystemSize) {
auto it = BaseType::mDoFSlaveSet.find(dof);
if (it == BaseType::mDoFSlaveSet.end()) {
++counter;
}
}
}

BaseType::mDoFToSolveSystemSize = counter;

KRATOS_CATCH("ContactResidualBasedEliminationBuilderAndSolverWithConstraints::FormulateGlobalMasterSlaveRelations failed ..");
}


void BaseSetUpSystem(ModelPart& rModelPart)
{


std::unordered_map<IndexType, IndexSetType> set_nodes_with_lm_associated;
if (rModelPart.HasSubModelPart("Contact"))
set_nodes_with_lm_associated.reserve(rModelPart.GetSubModelPart("Contact").NumberOfNodes());
IndexType node_id;
for (auto& i_dof : BaseType::mDofSet) {
node_id = i_dof.Id();
if (IsLMDof(i_dof))
set_nodes_with_lm_associated.insert({node_id, IndexSetType({})});
}

const IndexType key_lm_x = VECTOR_LAGRANGE_MULTIPLIER_X.Key();
const IndexType key_lm_y = VECTOR_LAGRANGE_MULTIPLIER_Y.Key();
const IndexType key_lm_z = VECTOR_LAGRANGE_MULTIPLIER_Z.Key();

for (auto& i_dof : BaseType::mDofSet) {
node_id = i_dof.Id();
auto it = set_nodes_with_lm_associated.find(node_id);
if ( it != set_nodes_with_lm_associated.end()) {
if (i_dof.IsFixed()) {
const auto& r_variable = i_dof.GetVariable();
auto& aux_set = (it->second);
if (r_variable == DISPLACEMENT_X) {
aux_set.insert(key_lm_x);
} else if (r_variable == DISPLACEMENT_Y) {
aux_set.insert(key_lm_y);
} else if (r_variable == DISPLACEMENT_Z) {
aux_set.insert(key_lm_z);
}
}
}
}

for (auto& i_dof : BaseType::mDofSet) {
if (i_dof.IsFree()) {
node_id = i_dof.Id();
auto it = set_nodes_with_lm_associated.find(node_id);
if (it != set_nodes_with_lm_associated.end()) {
auto& aux_set = it->second;
if (aux_set.find((i_dof.GetVariable()).Key()) != aux_set.end()) {
i_dof.FixDof();
}
}
}
}

BaseType::SetUpSystem(rModelPart);
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





}; 





} 