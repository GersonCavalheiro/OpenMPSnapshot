
#pragma once



#include "solving_strategies/builder_and_solvers/residualbased_block_builder_and_solver.h"

namespace Kratos
{







template<class TSparseSpace,
class TDenseSpace, 
class TLinearSolver, 
class TBuilderAndSolver = ResidualBasedBlockBuilderAndSolver< TSparseSpace, TDenseSpace, TLinearSolver >
>
class ContactResidualBasedBlockBuilderAndSolver
: public TBuilderAndSolver
{
public:

KRATOS_CLASS_POINTER_DEFINITION(ContactResidualBasedBlockBuilderAndSolver);

typedef BuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver> BaseBuilderAndSolverType;

typedef TBuilderAndSolver BaseType;

typedef ContactResidualBasedBlockBuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver, TBuilderAndSolver> ClassType;

typedef typename BaseType::TSchemeType TSchemeType;

typedef typename BaseType::TDataType TDataType;

typedef typename BaseType::DofsArrayType DofsArrayType;

typedef typename BaseType::TSystemMatrixType TSystemMatrixType;

typedef typename BaseType::TSystemVectorType TSystemVectorType;



explicit ContactResidualBasedBlockBuilderAndSolver() : BaseType()
{
}


explicit ContactResidualBasedBlockBuilderAndSolver(
typename TLinearSolver::Pointer pNewLinearSystemSolver,
Parameters ThisParameters
) : BaseType(pNewLinearSystemSolver)
{
ThisParameters = this->ValidateAndAssignParameters(ThisParameters, this->GetDefaultParameters());
this->AssignSettings(ThisParameters);
}


ContactResidualBasedBlockBuilderAndSolver(
typename TLinearSolver::Pointer pNewLinearSystemSolver)
: BaseType(pNewLinearSystemSolver)
{
}


~ContactResidualBasedBlockBuilderAndSolver() override
{
}




typename BaseBuilderAndSolverType::Pointer Create(
typename TLinearSolver::Pointer pNewLinearSystemSolver,
Parameters ThisParameters
) const override
{
return Kratos::make_shared<ClassType>(pNewLinearSystemSolver,ThisParameters);
}


void ApplyDirichletConditions(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemMatrixType& A,
TSystemVectorType& Dx,
TSystemVectorType& b
) override
{
FixIsolatedNodes(rModelPart);

BaseType::ApplyDirichletConditions(pScheme, rModelPart, A, Dx, b);

FreeIsolatedNodes(rModelPart);
}


void BuildRHS(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemVectorType& b
) override
{
FixIsolatedNodes(rModelPart);

BaseType::BuildRHS(pScheme, rModelPart, b);

FreeIsolatedNodes(rModelPart);
}


Parameters GetDefaultParameters() const override
{
Parameters default_parameters = Parameters(R"(
{
"name" : "contact_block_builder_and_solver"
})");

const Parameters base_default_parameters = BaseType::GetDefaultParameters();
default_parameters.RecursivelyAddMissingParameters(base_default_parameters);
return default_parameters;
}


static std::string Name()
{
return "contact_block_builder_and_solver";
}





protected:





void AssignSettings(const Parameters ThisParameters) override
{
BaseType::AssignSettings(ThisParameters);
}





private:




void FixIsolatedNodes(ModelPart& rModelPart)
{
KRATOS_ERROR_IF_NOT(rModelPart.HasSubModelPart("Contact")) << "CONTACT MODEL PART NOT CREATED" << std::endl;
KRATOS_ERROR_IF_NOT(rModelPart.HasSubModelPart("ComputingContact")) << "CONTACT COMPUTING MODEL PART NOT CREATED" << std::endl;
ModelPart& r_contact_model_part = rModelPart.GetSubModelPart("Contact");
ModelPart& r_computing_contact_model_part = rModelPart.GetSubModelPart("ComputingContact");

auto& r_nodes_array = r_contact_model_part.Nodes();
block_for_each(r_nodes_array, [&](NodeType& rNode) {
rNode.Set(VISITED, false);
rNode.Set(ISOLATED, false);
});

auto& r_conditions_array = r_computing_contact_model_part.Conditions();
block_for_each(r_conditions_array, [&](Condition& rCond) {
auto& r_parent_geometry = rCond.GetGeometry().GetGeometryPart(0);
for (std::size_t i_node = 0; i_node < r_parent_geometry.size(); ++i_node) {
r_parent_geometry[i_node].SetLock();
if (r_parent_geometry[i_node].Is(VISITED) == false) {
r_parent_geometry[i_node].Set(ISOLATED, rCond.Is(ISOLATED));
r_parent_geometry[i_node].Set(VISITED, true);
} else {
r_parent_geometry[i_node].Set(ISOLATED, r_parent_geometry[i_node].Is(ISOLATED) && rCond.Is(ISOLATED));
}
r_parent_geometry[i_node].UnSetLock();
}
});

block_for_each(r_nodes_array, [&](NodeType& rNode) {
if (rNode.Is(ISOLATED)) {
if (rNode.SolutionStepsDataHas(LAGRANGE_MULTIPLIER_CONTACT_PRESSURE)) {
rNode.Fix(LAGRANGE_MULTIPLIER_CONTACT_PRESSURE);
} else if (rNode.SolutionStepsDataHas(VECTOR_LAGRANGE_MULTIPLIER_X)) {
rNode.Fix(VECTOR_LAGRANGE_MULTIPLIER_X);
rNode.Fix(VECTOR_LAGRANGE_MULTIPLIER_Y);
rNode.Fix(VECTOR_LAGRANGE_MULTIPLIER_Z);
}
}
});
}


void FreeIsolatedNodes(ModelPart& rModelPart)
{
KRATOS_ERROR_IF_NOT(rModelPart.HasSubModelPart("Contact")) << "CONTACT MODEL PART NOT CREATED" << std::endl;
ModelPart& r_contact_model_part = rModelPart.GetSubModelPart("Contact");

auto& r_nodes_array = r_contact_model_part.Nodes();
block_for_each(r_nodes_array, [&](NodeType& rNode) {
if (rNode.Is(ISOLATED)) {
if (rNode.SolutionStepsDataHas(LAGRANGE_MULTIPLIER_CONTACT_PRESSURE)) {
rNode.Free(LAGRANGE_MULTIPLIER_CONTACT_PRESSURE);
} else if (rNode.SolutionStepsDataHas(VECTOR_LAGRANGE_MULTIPLIER_X)) {
rNode.Free(VECTOR_LAGRANGE_MULTIPLIER_X);
rNode.Free(VECTOR_LAGRANGE_MULTIPLIER_Y);
rNode.Free(VECTOR_LAGRANGE_MULTIPLIER_Z);
}
}
});
}






}; 





} 
