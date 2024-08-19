
#pragma once

#include <unordered_map>
#include <variant>
#include <vector>

#include "includes/define.h"
#include "includes/model_part.h"


namespace Kratos
{


class KRATOS_API(OPTIMIZATION_APPLICATION) MaxOverhangAngleResponseUtils
{
public:

using IndexType = std::size_t;

using GeometryType = ModelPart::ElementType::GeometryType;

using array_3d = array_1d<double, 3>;

using SensitivityFieldVariableTypes = std::variant<const Variable<double>*, const Variable<array_3d>*>;

using SensitivityVariableModelPartsListMap = std::unordered_map<SensitivityFieldVariableTypes, std::vector<ModelPart*>>;


static double CalculateValue(const std::vector<ModelPart*>& rModelParts, 
const Parameters ResponseSettings);

static void CalculateSensitivity(
const std::vector<ModelPart*>& rEvaluatedModelParts,
const SensitivityVariableModelPartsListMap& rSensitivityVariableModelPartInfo,
const Parameters ResponseSettings);

private:

static double CalculateConditionValue(const Condition& rCondition, const Parameters ResponseSettings);

static void CalculateConditionFiniteDifferenceShapeSensitivity(
Condition& rCondition,
Condition::Pointer& pThreadLocalCondition,
ModelPart& rModelPart,
std::vector<std::string>& rModelPartNames,
const Parameters ResponseSettings,
const IndexType MaxNodeId,
const Variable<array_3d>& rOutputSensitivityVariable);

static void CalculateFiniteDifferenceShapeSensitivity(
ModelPart& rModelPart,
const Parameters ResponseSettings,
const Variable<array_3d>& rOutputSensitivityVariable);

};

}