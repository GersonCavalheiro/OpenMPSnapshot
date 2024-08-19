
#pragma once




#include "adjoint_structural_response_function.h"
#include "stress_response_definitions.h"

namespace Kratos
{







class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) AdjointMaxStressResponseFunction : public AdjointStructuralResponseFunction
{
public:

KRATOS_CLASS_POINTER_DEFINITION(AdjointMaxStressResponseFunction);


AdjointMaxStressResponseFunction(ModelPart& rModelPart, Parameters ResponseSettings);

~AdjointMaxStressResponseFunction() override;



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


ModelPart& mrAdjointModelPart;
std::string mCriticalPartName;
Element::Pointer mpTracedElementInAdjointPart = nullptr;
StressTreatment mStressTreatment;
TracedStressType mTracedStressType;
SizeType mEchoLevel = 0;




void CalculateElementContributionToPartialSensitivity(Element& rAdjointElement,
const std::string& rVariableName,
const Matrix& rSensitivityMatrix,
Vector& rSensitivityGradient,
const ProcessInfo& rProcessInfo);

void ExtractMeanStressDerivative(const Matrix& rStressDerivativesMatrix, Vector& rResponseGradient);





}; 





} 
