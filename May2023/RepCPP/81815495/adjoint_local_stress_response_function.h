
#pragma once




#include "adjoint_structural_response_function.h"
#include "stress_response_definitions.h"

namespace Kratos
{







class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) AdjointLocalStressResponseFunction : public AdjointStructuralResponseFunction
{
public:

typedef Element::DofsVectorType DofsVectorType;
typedef Variable<double>* Array1DComponentsPointerType;

KRATOS_CLASS_POINTER_DEFINITION(AdjointLocalStressResponseFunction);


AdjointLocalStressResponseFunction(ModelPart& rModelPart, Parameters ResponseSettings);

~AdjointLocalStressResponseFunction() override;



void FinalizeSolutionStep() override;

using AdjointStructuralResponseFunction::CalculateGradient;

void CalculateGradient(const Element& rAdjointElement,
const Matrix& rResidualGradient,
Vector& rResponseGradient,
const ProcessInfo& rProcessInfo) override;

void CalculatePartialSensitivity(Element& rAdjointElement,
const Variable<double>& rVariable,
const Matrix& rSensitivityMatrix,
Vector& rSensitivityGradient,
const ProcessInfo& rProcessInfo) override;

void CalculatePartialSensitivity(Condition& rAdjointCondition,
const Variable<double>& rVariable,
const Matrix& rSensitivityMatrix,
Vector& rSensitivityGradient,
const ProcessInfo& rProcessInfo) override;

void CalculatePartialSensitivity(Element& rAdjointElement,
const Variable<array_1d<double, 3>>& rVariable,
const Matrix& rSensitivityMatrix,
Vector& rSensitivityGradient,
const ProcessInfo& rProcessInfo) override;

void CalculatePartialSensitivity(Condition& rAdjointCondition,
const Variable<array_1d<double, 3>>& rVariable,
const Matrix& rSensitivityMatrix,
Vector& rSensitivityGradient,
const ProcessInfo& rProcessInfo) override;

double CalculateValue(ModelPart& rModelPart) override;







protected:








private:


unsigned int mIdOfLocation;
Element::Pointer mpTracedElement;
StressTreatment mStressTreatment;
TracedStressType mTracedStressType;
bool mAddParticularSolution = false;




double CalculateMeanElementStress(ModelPart& rModelPart);

double CalculateGaussPointStress(ModelPart& rModelPart);

double CalculateNodeStress(ModelPart& rModelPart);

void CalculateElementContributionToPartialSensitivity(Element& rAdjointElement,
const std::string& rVariableName,
const Matrix& rSensitivityMatrix,
Vector& rSensitivityGradient,
const ProcessInfo& rProcessInfo);

void ExtractMeanStressDerivative(const Matrix& rStressDerivativesMatrix, Vector& rResponseGradient);

void ExtractNodeStressDerivative(const Matrix& rStressDerivativesMatrix, Vector& rResponseGradient);

void ExtractGaussPointStressDerivative(const Matrix& rStressDerivativesMatrix, Vector& rResponseGradient);

void CalculateParticularSolution() const;

void CalculateParticularSolutionLinearElement2N(Vector& rResult) const;

void CalculateMeanParticularSolutionLinearElement2N(Vector& rResult, DofsVectorType &rElementalDofList, const Array1DComponentsPointerType TracedDof) const;

void CalculateGPParticularSolutionLinearElement2N(Vector& rResult, DofsVectorType &rElementalDofList, const Array1DComponentsPointerType TracedDof) const;

void CalculateNodeParticularSolutionLinearElement2N(Vector& rResult, DofsVectorType &rElementalDofList, const Array1DComponentsPointerType TracedDof) const;

void FindVariableComponent(Array1DComponentsPointerType& rTracedDof) const;





}; 





} 
