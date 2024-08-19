
#pragma once



#include "includes/condition.h"

namespace Kratos
{








template <typename TPrimalCondition>
class AdjointSemiAnalyticBaseCondition
: public Condition
{
public:
KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION( AdjointSemiAnalyticBaseCondition );


AdjointSemiAnalyticBaseCondition(IndexType NewId = 0)
: Condition(NewId),
mpPrimalCondition(Kratos::make_intrusive<TPrimalCondition>(NewId, pGetGeometry()))
{
}

AdjointSemiAnalyticBaseCondition(IndexType NewId, GeometryType::Pointer pGeometry)
: Condition(NewId, pGeometry),
mpPrimalCondition(Kratos::make_intrusive<TPrimalCondition>(NewId, pGeometry))
{
}

AdjointSemiAnalyticBaseCondition(IndexType NewId,
GeometryType::Pointer pGeometry,
PropertiesType::Pointer pProperties)
: Condition(NewId, pGeometry, pProperties),
mpPrimalCondition(Kratos::make_intrusive<TPrimalCondition>(NewId, pGeometry, pProperties))
{
}




Condition::Pointer Create(IndexType NewId,
NodesArrayType const& ThisNodes,
PropertiesType::Pointer pProperties) const override
{
return Kratos::make_intrusive<AdjointSemiAnalyticBaseCondition<TPrimalCondition>>(
NewId, GetGeometry().Create(ThisNodes), pProperties);
}

Condition::Pointer Create(IndexType NewId,
GeometryType::Pointer pGeometry,
PropertiesType::Pointer pProperties) const override
{
return Kratos::make_intrusive<AdjointSemiAnalyticBaseCondition<TPrimalCondition>>(
NewId, pGeometry, pProperties);
}

void EquationIdVector(EquationIdVectorType& rResult, const ProcessInfo& rCurrentProcessInfo ) const override;

void GetDofList(DofsVectorType& ElementalDofList, const ProcessInfo& rCurrentProcessInfo ) const override;

IntegrationMethod GetIntegrationMethod() const override
{
return mpPrimalCondition->GetIntegrationMethod();
}

void GetValuesVector(Vector& rValues, int Step = 0 ) const override;

void Initialize(const ProcessInfo& rCurrentProcessInfo) override
{
mpPrimalCondition->Initialize(rCurrentProcessInfo);
}

void ResetConstitutiveLaw() override
{
mpPrimalCondition->ResetConstitutiveLaw();
}

void InitializeSolutionStep(const ProcessInfo& rCurrentProcessInfo) override
{
mpPrimalCondition->InitializeSolutionStep(rCurrentProcessInfo);
}

void InitializeNonLinearIteration(const ProcessInfo& rCurrentProcessInfo) override
{
mpPrimalCondition->InitializeNonLinearIteration(rCurrentProcessInfo);
}

void FinalizeNonLinearIteration(const ProcessInfo& rCurrentProcessInfo) override
{
mpPrimalCondition->FinalizeNonLinearIteration(rCurrentProcessInfo);
}

void FinalizeSolutionStep(const ProcessInfo& rCurrentProcessInfo) override
{
mpPrimalCondition->FinalizeSolutionStep(rCurrentProcessInfo);
}

void CalculateLocalSystem(MatrixType& rLeftHandSideMatrix,
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo) override
{
mpPrimalCondition->CalculateLocalSystem(rLeftHandSideMatrix,
rRightHandSideVector,
rCurrentProcessInfo);
}

void CalculateLeftHandSide(MatrixType& rLeftHandSideMatrix,
const ProcessInfo& rCurrentProcessInfo) override
{
mpPrimalCondition->CalculateLeftHandSide(rLeftHandSideMatrix,
rCurrentProcessInfo);
}

void CalculateRightHandSide(VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo) override
{
mpPrimalCondition->CalculateRightHandSide(rRightHandSideVector,
rCurrentProcessInfo);
}

void CalculateFirstDerivativesContributions(MatrixType& rLeftHandSideMatrix,
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo) override
{
mpPrimalCondition->CalculateFirstDerivativesContributions(rLeftHandSideMatrix,
rRightHandSideVector,
rCurrentProcessInfo);
}

void CalculateFirstDerivativesLHS(MatrixType& rLeftHandSideMatrix,
const ProcessInfo& rCurrentProcessInfo) override
{
mpPrimalCondition->CalculateFirstDerivativesLHS(rLeftHandSideMatrix,
rCurrentProcessInfo);
}

void CalculateFirstDerivativesRHS(VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo) override
{
mpPrimalCondition->CalculateFirstDerivativesRHS(rRightHandSideVector,
rCurrentProcessInfo);
}

void CalculateSecondDerivativesContributions(MatrixType& rLeftHandSideMatrix,
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo) override
{
mpPrimalCondition->CalculateSecondDerivativesContributions(rLeftHandSideMatrix,
rRightHandSideVector,
rCurrentProcessInfo);
}

void CalculateSecondDerivativesLHS(MatrixType& rLeftHandSideMatrix,
const ProcessInfo& rCurrentProcessInfo) override
{
mpPrimalCondition->CalculateSecondDerivativesLHS(rLeftHandSideMatrix,
rCurrentProcessInfo);
}

void CalculateSecondDerivativesRHS(VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo) override
{
mpPrimalCondition->CalculateSecondDerivativesRHS(rRightHandSideVector,
rCurrentProcessInfo);
}

void CalculateMassMatrix(MatrixType& rMassMatrix, const ProcessInfo& rCurrentProcessInfo) override
{
mpPrimalCondition->CalculateMassMatrix(rMassMatrix, rCurrentProcessInfo);
}

void CalculateDampingMatrix(MatrixType& rDampingMatrix, const ProcessInfo& rCurrentProcessInfo) override
{
mpPrimalCondition->CalculateDampingMatrix(rDampingMatrix, rCurrentProcessInfo);
}

void Calculate(const Variable<double >& rVariable,
double& Output,
const ProcessInfo& rCurrentProcessInfo) override
{
KRATOS_ERROR << "Calculate of the adjoint base condition is called!" << std::endl;
}

void Calculate(const Variable< array_1d<double,3> >& rVariable,
array_1d<double,3>& Output,
const ProcessInfo& rCurrentProcessInfo) override
{
KRATOS_ERROR << "Calculate of the adjoint base condition is called!" << std::endl;
}

void Calculate(const Variable<Vector >& rVariable,
Vector& Output,
const ProcessInfo& rCurrentProcessInfo) override
{
KRATOS_ERROR << "Calculate of the adjoint base condition is called!" << std::endl;
}

void Calculate(const Variable<Matrix >& rVariable,
Matrix& Output,
const ProcessInfo& rCurrentProcessInfo) override
{
KRATOS_ERROR << "Calculate of the adjoint base condition is called!" << std::endl;
}


void CalculateOnIntegrationPoints(const Variable<double>& rVariable,
std::vector<double>& rOutput,
const ProcessInfo& rCurrentProcessInfo) override;

void CalculateOnIntegrationPoints(const Variable<array_1d<double, 3 > >& rVariable,
std::vector< array_1d<double, 3 > >& Output,
const ProcessInfo& rCurrentProcessInfo) override;

void CalculateOnIntegrationPoints(const Variable<Vector >& rVariable,
std::vector< Vector >& Output,
const ProcessInfo& rCurrentProcessInfo) override
{
KRATOS_ERROR << "CalculateOnIntegrationPoints of the adjoint base condition is called!" << std::endl;
}

void CalculateOnIntegrationPoints(const Variable<Matrix >& rVariable,
std::vector< Matrix >& Output,
const ProcessInfo& rCurrentProcessInfo) override
{
KRATOS_ERROR << "CalculateOnIntegrationPoints of the adjoint base condition is called!" << std::endl;
}

int Check( const ProcessInfo& rCurrentProcessInfo ) const override;


void CalculateSensitivityMatrix(const Variable<double>& rDesignVariable,
Matrix& rOutput,
const ProcessInfo& rCurrentProcessInfo) override;


void CalculateSensitivityMatrix(const Variable<array_1d<double,3> >& rDesignVariable,
Matrix& rOutput,
const ProcessInfo& rCurrentProcessInfo) override;

Condition::Pointer pGetPrimalCondition()
{
return mpPrimalCondition;
}

const Condition::Pointer pGetPrimalCondition() const
{
return mpPrimalCondition;
}










protected:



Condition::Pointer mpPrimalCondition;










private:








double GetPerturbationSize(const Variable<double>& rDesignVariable, const ProcessInfo& rCurrentProcessInfo) const;


double GetPerturbationSize(const Variable<array_1d<double,3>>& rDesignVariable, const ProcessInfo& rCurrentProcessInfo) const;


virtual double GetPerturbationSizeModificationFactor(const Variable<double>& rVariable) const;


virtual double GetPerturbationSizeModificationFactor(const Variable<array_1d<double,3>>& rDesignVariable) const;






friend class Serializer;

void save( Serializer& rSerializer ) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS( rSerializer, Condition );
rSerializer.save("mpPrimalCondition", mpPrimalCondition);
}

void load( Serializer& rSerializer ) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS( rSerializer, Condition );
rSerializer.load("mpPrimalCondition", mpPrimalCondition);
}



}; 





}  


