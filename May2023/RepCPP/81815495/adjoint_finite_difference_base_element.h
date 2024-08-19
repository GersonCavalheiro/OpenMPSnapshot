
#pragma once




#include "includes/element.h"
#include "structural_mechanics_application_variables.h"
#include "utilities/openmp_utils.h"

namespace Kratos
{







template <typename TPrimalElement>
class AdjointFiniteDifferencingBaseElement : public Element
{
public:


KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION(AdjointFiniteDifferencingBaseElement);



AdjointFiniteDifferencingBaseElement(IndexType NewId = 0,
bool HasRotationDofs = false)
: Element(NewId),
mpPrimalElement(Kratos::make_intrusive<TPrimalElement>(NewId, pGetGeometry())),
mHasRotationDofs(HasRotationDofs)
{
}

AdjointFiniteDifferencingBaseElement(IndexType NewId,
GeometryType::Pointer pGeometry,
bool HasRotationDofs = false)
: Element(NewId, pGeometry),
mpPrimalElement(Kratos::make_intrusive<TPrimalElement>(NewId, pGeometry)),
mHasRotationDofs(HasRotationDofs)
{
}

AdjointFiniteDifferencingBaseElement(IndexType NewId,
GeometryType::Pointer pGeometry,
PropertiesType::Pointer pProperties,
bool HasRotationDofs = false)
: Element(NewId, pGeometry, pProperties),
mpPrimalElement(Kratos::make_intrusive<TPrimalElement>(NewId, pGeometry, pProperties)),
mHasRotationDofs(HasRotationDofs)
{
}




Element::Pointer Create(IndexType NewId,
NodesArrayType const& ThisNodes,
PropertiesType::Pointer pProperties) const override
{
return Kratos::make_intrusive<AdjointFiniteDifferencingBaseElement<TPrimalElement>>(
NewId, GetGeometry().Create(ThisNodes), pProperties);
}

Element::Pointer Create(IndexType NewId,
GeometryType::Pointer pGeometry,
PropertiesType::Pointer pProperties) const override
{
return Kratos::make_intrusive<AdjointFiniteDifferencingBaseElement<TPrimalElement>>(
NewId, pGeometry, pProperties);
}

void EquationIdVector(EquationIdVectorType& rResult, const ProcessInfo& rCurrentProcessInfo) const override;

void GetDofList(DofsVectorType& ElementalDofList, const ProcessInfo& CurrentProcessInfo) const override;


IntegrationMethod GetIntegrationMethod() const override
{
return mpPrimalElement->GetIntegrationMethod();
}

void GetValuesVector(Vector& values, int Step = 0) const override;

void Initialize(const ProcessInfo& rCurrentProcessInfo) override
{
mpPrimalElement->Initialize(rCurrentProcessInfo);
}

void ResetConstitutiveLaw() override
{
mpPrimalElement->ResetConstitutiveLaw();
}

void InitializeSolutionStep(const ProcessInfo& rCurrentProcessInfo) override
{
mpPrimalElement->InitializeSolutionStep(rCurrentProcessInfo);
}

void InitializeNonLinearIteration(const ProcessInfo& rCurrentProcessInfo) override
{
mpPrimalElement->InitializeNonLinearIteration(rCurrentProcessInfo);
}

void FinalizeNonLinearIteration(const ProcessInfo& rCurrentProcessInfo) override
{
mpPrimalElement->FinalizeNonLinearIteration(rCurrentProcessInfo);
}

void FinalizeSolutionStep(const ProcessInfo& rCurrentProcessInfo) override
{
mpPrimalElement->FinalizeSolutionStep(rCurrentProcessInfo);
}

void CalculateLocalSystem(MatrixType& rLeftHandSideMatrix,
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo) override
{
mpPrimalElement->CalculateLocalSystem(rLeftHandSideMatrix,
rRightHandSideVector,
rCurrentProcessInfo);
}

void CalculateLeftHandSide(MatrixType& rLeftHandSideMatrix,
const ProcessInfo& rCurrentProcessInfo) override
{
mpPrimalElement->CalculateLeftHandSide(rLeftHandSideMatrix,
rCurrentProcessInfo);
}

void CalculateRightHandSide(VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo) override
{
mpPrimalElement->CalculateRightHandSide(rRightHandSideVector,
rCurrentProcessInfo);
}

void CalculateFirstDerivativesContributions(MatrixType& rLeftHandSideMatrix,
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo) override
{
mpPrimalElement->CalculateFirstDerivativesContributions(rLeftHandSideMatrix,
rRightHandSideVector,
rCurrentProcessInfo);
}

void CalculateFirstDerivativesLHS(MatrixType& rLeftHandSideMatrix,
const ProcessInfo& rCurrentProcessInfo) override
{
mpPrimalElement->CalculateFirstDerivativesLHS(rLeftHandSideMatrix,
rCurrentProcessInfo);
}

void CalculateFirstDerivativesRHS(VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo) override
{
mpPrimalElement->CalculateFirstDerivativesRHS(rRightHandSideVector,
rCurrentProcessInfo);
}

void CalculateSecondDerivativesContributions(MatrixType& rLeftHandSideMatrix,
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo) override
{
mpPrimalElement->CalculateSecondDerivativesContributions(rLeftHandSideMatrix,
rRightHandSideVector,
rCurrentProcessInfo);
}

void CalculateSecondDerivativesLHS(MatrixType& rLeftHandSideMatrix,
const ProcessInfo& rCurrentProcessInfo) override
{
mpPrimalElement->CalculateSecondDerivativesLHS(rLeftHandSideMatrix,
rCurrentProcessInfo);
}

void CalculateSecondDerivativesRHS(VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo) override
{
mpPrimalElement->CalculateSecondDerivativesRHS(rRightHandSideVector,
rCurrentProcessInfo);
}

void CalculateMassMatrix(MatrixType& rMassMatrix, const ProcessInfo& rCurrentProcessInfo) override
{
mpPrimalElement->CalculateMassMatrix(rMassMatrix,rCurrentProcessInfo);
}

void CalculateDampingMatrix(MatrixType& rDampingMatrix, const ProcessInfo& rCurrentProcessInfo) override
{
mpPrimalElement->CalculateDampingMatrix(rDampingMatrix, rCurrentProcessInfo);
}

void AddExplicitContribution(const ProcessInfo& rCurrentProcessInfo) override
{
mpPrimalElement->AddExplicitContribution(rCurrentProcessInfo);
}

void AddExplicitContribution(const VectorType& rRHSVector,
const Variable<VectorType>& rRHSVariable,
const Variable<double >& rDestinationVariable,
const ProcessInfo& rCurrentProcessInfo) override
{
mpPrimalElement->AddExplicitContribution(rRHSVector,
rRHSVariable,
rDestinationVariable,
rCurrentProcessInfo);
}

void AddExplicitContribution(const VectorType& rRHSVector,
const Variable<VectorType>& rRHSVariable,
const Variable<array_1d<double,3> >& rDestinationVariable,
const ProcessInfo& rCurrentProcessInfo) override
{
mpPrimalElement->AddExplicitContribution(rRHSVector,
rRHSVariable,
rDestinationVariable,
rCurrentProcessInfo);
}

void AddExplicitContribution(const MatrixType& rLHSMatrix,
const Variable<MatrixType>& rLHSVariable,
const Variable<Matrix>& rDestinationVariable,
const ProcessInfo& rCurrentProcessInfo) override
{
mpPrimalElement->AddExplicitContribution(rLHSMatrix,
rLHSVariable,
rDestinationVariable,
rCurrentProcessInfo);
}

void Calculate(const Variable<double >& rVariable,
double& Output,
const ProcessInfo& rCurrentProcessInfo) override
{
KRATOS_ERROR << "Calculate of the adjoint base element is called!" << std::endl;
}

void Calculate(const Variable< array_1d<double,3> >& rVariable,
array_1d<double,3>& Output,
const ProcessInfo& rCurrentProcessInfo) override
{
KRATOS_ERROR << "Calculate of the adjoint base element is called!" << std::endl;
}

void Calculate(const Variable<Vector >& rVariable,
Vector& Output,
const ProcessInfo& rCurrentProcessInfo) override
{
KRATOS_ERROR << "Calculate of the adjoint base element is called!" << std::endl;
}

void Calculate(const Variable<Matrix >& rVariable,
Matrix& Output,
const ProcessInfo& rCurrentProcessInfo) override;

void CalculateOnIntegrationPoints(const Variable<bool>& rVariable,
std::vector<bool>& rOutput,
const ProcessInfo& rCurrentProcessInfo) override
{
KRATOS_ERROR << "CalculateOnIntegrationPoints of the adjoint base element is called!" << std::endl;
}

void CalculateOnIntegrationPoints(const Variable<double>& rVariable,
std::vector<double>& rOutput,
const ProcessInfo& rCurrentProcessInfo) override;

void CalculateOnIntegrationPoints(const Variable<array_1d<double, 3 > >& rVariable,
std::vector< array_1d<double, 3 > >& rOutput,
const ProcessInfo& rCurrentProcessInfo) override;

void CalculateOnIntegrationPoints(const Variable<array_1d<double, 6 > >& rVariable,
std::vector< array_1d<double, 6 > >& rOutput,
const ProcessInfo& rCurrentProcessInfo) override
{
KRATOS_ERROR << "CalculateOnIntegrationPoints of the adjoint base element is called!" << std::endl;
}

void CalculateOnIntegrationPoints(const Variable<Vector >& rVariable,
std::vector< Vector >& rOutput,
const ProcessInfo& rCurrentProcessInfo) override
{
KRATOS_ERROR << "CalculateOnIntegrationPoints of the adjoint base element is called!" << std::endl;
}

void CalculateOnIntegrationPoints(const Variable<Matrix >& rVariable,
std::vector< Matrix >& rOutput,
const ProcessInfo& rCurrentProcessInfo) override
{
KRATOS_ERROR << "CalculateOnIntegrationPoints of the adjoint base element is called!" << std::endl;
}


int Check(const ProcessInfo& rCurrentProcessInfo) const override;



void CalculateSensitivityMatrix(const Variable<double>& rDesignVariable, Matrix& rOutput,
const ProcessInfo& rCurrentProcessInfo) override;


void CalculateSensitivityMatrix(const Variable<array_1d<double,3>>& rDesignVariable, Matrix& rOutput,
const ProcessInfo& rCurrentProcessInfo) override;


virtual void CalculateStressDisplacementDerivative(const Variable<Vector>& rStressVariable,
Matrix& rOutput, const ProcessInfo& rCurrentProcessInfo);

void CalculateStressDesignVariableDerivative(const Variable<double>& rDesignVariable, const Variable<Vector>& rStressVariable,
Matrix& rOutput, const ProcessInfo& rCurrentProcessInfo);

void CalculateStressDesignVariableDerivative(const Variable<array_1d<double,3>>& rDesignVariable,
const Variable<Vector>& rStressVariable,
Matrix& rOutput, const ProcessInfo& rCurrentProcessInfo);

Element::Pointer pGetPrimalElement()
{
return mpPrimalElement;
}
const Element::Pointer pGetPrimalElement() const
{
return mpPrimalElement;
}


protected:




template <typename TDataType>
void CalculateAdjointFieldOnIntegrationPoints(const Variable<TDataType>& rVariable, std::vector< TDataType >& rOutput, const ProcessInfo& rCurrentProcessInfo)
{
KRATOS_TRY;

KRATOS_WARNING_IF("CalculateAdjointFieldOnIntegrationPoints", OpenMPUtils::IsInParallel() != 0)
<< "The call of this non shared-memory-parallelized function within a parallel section should be avoided for efficiency reasons!" << std::endl;

const SizeType num_nodes = mpPrimalElement->GetGeometry().PointsNumber();
const SizeType dimension = mpPrimalElement->GetGeometry().WorkingSpaceDimension();
const SizeType num_dofs_per_node = (mHasRotationDofs) ?  2 * dimension : dimension;
const SizeType num_dofs = num_nodes * num_dofs_per_node;
Vector initial_state_variables;
initial_state_variables.resize(num_dofs);

Vector particular_solution = ZeroVector(num_dofs);
if(this->Has(ADJOINT_PARTICULAR_DISPLACEMENT)) {
particular_solution = this->GetValue(ADJOINT_PARTICULAR_DISPLACEMENT);
}

std::vector<Variable<double>*> primal_solution_variable_list;
(mHasRotationDofs) ? primal_solution_variable_list = {&DISPLACEMENT_X, &DISPLACEMENT_Y, &DISPLACEMENT_Z, &ROTATION_X, &ROTATION_Y, &ROTATION_Z} :
primal_solution_variable_list = {&DISPLACEMENT_X, &DISPLACEMENT_Y, &DISPLACEMENT_Z};

std::vector<Variable<double>*> adjoint_solution_variable_list;
(mHasRotationDofs) ? adjoint_solution_variable_list = {&ADJOINT_DISPLACEMENT_X, &ADJOINT_DISPLACEMENT_Y, &ADJOINT_DISPLACEMENT_Z, &ADJOINT_ROTATION_X, &ADJOINT_ROTATION_Y, &ADJOINT_ROTATION_Z} :
adjoint_solution_variable_list = {&ADJOINT_DISPLACEMENT_X, &ADJOINT_DISPLACEMENT_Y, &ADJOINT_DISPLACEMENT_Z};

for (IndexType i = 0; i < num_nodes; ++i) {
const IndexType index = i * num_dofs_per_node;
for(IndexType j = 0; j < primal_solution_variable_list.size(); ++j) {
initial_state_variables[index + j] = mpPrimalElement->GetGeometry()[i].FastGetSolutionStepValue(*primal_solution_variable_list[j]);
mpPrimalElement->GetGeometry()[i].FastGetSolutionStepValue(*primal_solution_variable_list[j]) =
this->GetGeometry()[i].FastGetSolutionStepValue(*adjoint_solution_variable_list[j]) +
particular_solution[index + j];
}
}

mpPrimalElement->CalculateOnIntegrationPoints(rVariable, rOutput, rCurrentProcessInfo);

for (IndexType i = 0; i < num_nodes; ++i) {
const IndexType index = i * num_dofs_per_node;
for(IndexType j = 0; j < primal_solution_variable_list.size(); ++j) {
mpPrimalElement->GetGeometry()[i].FastGetSolutionStepValue(*primal_solution_variable_list[j]) = initial_state_variables[index + j];
}
}

KRATOS_CATCH("")
}



Element::Pointer mpPrimalElement;
bool mHasRotationDofs = false;


private:




double GetPerturbationSize(const Variable<double>& rDesignVariable, const ProcessInfo& rCurrentProcessInfo) const;


double GetPerturbationSize(const Variable<array_1d<double,3>>& rDesignVariable, const ProcessInfo& rCurrentProcessInfo) const;


virtual double GetPerturbationSizeModificationFactor(const Variable<double>& rVariable) const;


virtual double GetPerturbationSizeModificationFactor(const Variable<array_1d<double,3>>& rDesignVariable) const;





friend class Serializer;

void save(Serializer& rSerializer) const override;

void load(Serializer& rSerializer) override;





};

}
