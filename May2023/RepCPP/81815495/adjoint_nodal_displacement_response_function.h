
#pragma once




#include "adjoint_structural_response_function.h"


namespace Kratos
{







class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) AdjointNodalDisplacementResponseFunction : public AdjointStructuralResponseFunction
{
public:

typedef Element::DofsVectorType DofsVectorType;
typedef Node::Pointer PointTypePointer;
typedef Variable<array_1d<double, 3>> ArrayVariableType;

KRATOS_CLASS_POINTER_DEFINITION(AdjointNodalDisplacementResponseFunction);


AdjointNodalDisplacementResponseFunction(ModelPart& rModelPart, Parameters ResponseSettings);

~AdjointNodalDisplacementResponseFunction() override;



using AdjointStructuralResponseFunction::CalculateGradient;

void CalculateGradient(const Element& rAdjointElement,
const Matrix& rResidualGradient,
Vector& rResponseGradient,
const ProcessInfo& rProcessInfo) override;

void CalculateFirstDerivativesGradient(const Element& rAdjointElement,
const Matrix& rResidualGradient,
Vector& rResponseGradient,
const ProcessInfo& rProcessInfo) override;

void CalculateFirstDerivativesGradient(const Condition& rAdjointCondition,
const Matrix& rResidualGradient,
Vector& rResponseGradient,
const ProcessInfo& rProcessInfo) override;

void CalculateSecondDerivativesGradient(const Element& rAdjointElement,
const Matrix& rResidualGradient,
Vector& rResponseGradient,
const ProcessInfo& rProcessInfo) override;

void CalculateSecondDerivativesGradient(const Condition& rAdjointCondition,
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



std::string mTracedDofLabel;
std::string mResponsePartName;
array_1d<double,3> mResponseDirection;
std::unordered_map<IndexType, std::vector<IndexType>> mElementNodeMap;



void ComputeNeighboringElementNodeMap();





}; 





} 
