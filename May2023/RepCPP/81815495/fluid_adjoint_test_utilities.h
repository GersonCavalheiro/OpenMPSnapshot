
#pragma once

#include <functional>
#include <vector>


#include "containers/model.h"
#include "includes/model_part.h"
#include "processes/process.h"


namespace Kratos
{

class KRATOS_API(FLUID_DYNAMICS_APPLICATION) FluidAdjointTestUtilities
{
public:

using IndexType = std::size_t;

using NodeType = ModelPart::NodeType;

using ConditionType = ModelPart::ConditionType;

using ElementType = ModelPart::ElementType;


template<class TDataType>
static TDataType CalculateRelaxedVariableRate(
const double BossakAlpha,
const Variable<TDataType>& rVariable,
const NodeType& rNode);

template<class TDataType>
static void RunAdjointEntityDerivativesTest(
ModelPart& rPrimalModelPart,
ModelPart& rAdjointModelPart,
const std::function<void(ModelPart&)>& rUpdateModelPart,
const Variable<TDataType>& rVariable,
const std::function<void(Matrix&, ConditionType&, const ProcessInfo&)>& rCalculateElementResidualDerivatives,
const IndexType EquationOffset,
const IndexType DerivativeOffset,
const double Delta,
const double Tolerance);

template<class TDataType>
static void RunAdjointEntityDerivativesTest(
ModelPart& rPrimalModelPart,
ModelPart& rAdjointModelPart,
const std::function<void(ModelPart&)>& rUpdateModelPart,
const Variable<TDataType>& rVariable,
const std::function<void(Matrix&, ElementType&, const ProcessInfo&)>& rCalculateElementResidualDerivatives,
const IndexType EquationOffset,
const IndexType DerivativeOffset,
const double Delta,
const double Tolerance);

};


} 