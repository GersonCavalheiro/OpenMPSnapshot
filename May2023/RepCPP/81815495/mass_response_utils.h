
#pragma once

#include <variant>
#include <vector>

#include "includes/define.h"
#include "includes/model_part.h"


namespace Kratos
{


class KRATOS_API(OPTIMIZATION_APPLICATION) MassResponseUtils
{
public:

using IndexType = std::size_t;

using GeometryType = ModelPart::ElementType::GeometryType;

using GradientFieldVariableTypes = std::variant<const Variable<double>*, const Variable<array_1d<double, 3>>*>;


static void Check(const ModelPart& rModelPart);

static double CalculateValue(const ModelPart& rModelPart);

static void CalculateGradient(
const std::vector<GradientFieldVariableTypes>& rListOfGradientVariables,
const std::vector<ModelPart*>& rListOfGradientRequiredModelParts,
const std::vector<ModelPart*>& rListOfGradientComputedModelParts);

private:

static bool HasVariableInProperties(
const ModelPart& rModelPart,
const Variable<double>& rVariable);

static void CalculateMassGeometricalPropertyGradient(
ModelPart& rModelPart,
const Variable<double>& rGeometricalPropertyGradientVariable,
const Variable<double>& rGeometricalCoflictingPropertyGradientVariable,
const Variable<double>& rOutputGradientVariable);

static void CalculateMassShapeGradient(
ModelPart& rModelPart,
const Variable<array_1d<double, 3>>& rOutputGradientVariable);

static void CalculateMassDensityGradient(
ModelPart& rModelPart,
const Variable<double>& rOutputGradientVariable);

static void CalculateMassThicknessGradient(
ModelPart& rModelPart,
const Variable<double>& rOutputGradientVariable);

static void CalculateMassCrossAreaGradient(
ModelPart& rModelPart,
const Variable<double>& rOutputGradientVariable);

};

}