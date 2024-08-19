
#pragma once

#include <unordered_set>
#include <unordered_map>


#include "solving_strategies/builder_and_solvers/residualbased_elimination_builder_and_solver.h"

namespace Kratos
{







template<class TSparseSpace,
class TDenseSpace, 
class TLinearSolver 
>
class ContactResidualBasedEliminationBuilderAndSolver
: public ResidualBasedEliminationBuilderAndSolver< TSparseSpace, TDenseSpace, TLinearSolver >
{
public:

KRATOS_CLASS_POINTER_DEFINITION(ContactResidualBasedEliminationBuilderAndSolver);

typedef BuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver> BaseBuilderAndSolverType;

typedef ResidualBasedEliminationBuilderAndSolver< TSparseSpace, TDenseSpace, TLinearSolver > BaseType;

typedef ContactResidualBasedEliminationBuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver> ClassType;

typedef typename BaseType::TSchemeType TSchemeType;
typedef typename BaseType::TDataType TDataType;
typedef typename BaseType::DofsArrayType DofsArrayType;
typedef typename BaseType::TSystemMatrixType TSystemMatrixType;
typedef typename BaseType::TSystemVectorType TSystemVectorType;

typedef Node NodeType;

typedef typename ModelPart::DofType DofType;

typedef std::size_t SizeType;

typedef std::size_t IndexType;

typedef std::unordered_set<IndexType> IndexSetType;





explicit ContactResidualBasedEliminationBuilderAndSolver() : BaseType()
{
}


explicit ContactResidualBasedEliminationBuilderAndSolver(
typename TLinearSolver::Pointer pNewLinearSystemSolver,
Parameters ThisParameters
) : BaseType(pNewLinearSystemSolver)
{
ThisParameters = this->ValidateAndAssignParameters(ThisParameters, this->GetDefaultParameters());
this->AssignSettings(ThisParameters);
}


ContactResidualBasedEliminationBuilderAndSolver(
typename TLinearSolver::Pointer pNewLinearSystemSolver)
: BaseType(pNewLinearSystemSolver)
{
}


~ContactResidualBasedEliminationBuilderAndSolver() override
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
const auto& r_variable = i_dof.GetVariable();
auto& aux_set = (it->second);
if (i_dof.IsFixed()) {
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


Parameters GetDefaultParameters() const override
{
Parameters default_parameters = Parameters(R"(
{
"name" : "contact_residual_elimination_builder_and_solver"
})");

const Parameters base_default_parameters = BaseType::GetDefaultParameters();
default_parameters.RecursivelyAddMissingParameters(base_default_parameters);
return default_parameters;
}


static std::string Name()
{
return "contact_residual_elimination_builder_and_solver";
}





protected:





void AssignSettings(const Parameters ThisParameters) override
{
BaseType::AssignSettings(ThisParameters);
}




private:





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