
#pragma once



#include "adjoint_structural_response_function.h"

namespace Kratos
{







class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) AdjointLinearStrainEnergyResponseFunction : public AdjointStructuralResponseFunction
{
public:

typedef AdjointStructuralResponseFunction BaseType;

KRATOS_CLASS_POINTER_DEFINITION(AdjointLinearStrainEnergyResponseFunction);


AdjointLinearStrainEnergyResponseFunction(ModelPart& rModelPart, Parameters ResponseSettings);

~AdjointLinearStrainEnergyResponseFunction();



void Initialize() override;

void CalculatePartialSensitivity(Element& rAdjointElement,
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
const Variable<double>& rVariable,
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




void CheckForBodyForces(Element& rAdjointElement);





}; 





} 
