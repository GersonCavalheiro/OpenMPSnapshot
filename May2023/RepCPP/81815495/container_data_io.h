
#pragma once

#include <string>

#include "containers/variable.h"
#include "includes/model_part.h"

namespace Kratos {


namespace ContainerDataIOTags {
struct Historical    {};
struct NonHistorical {};
} 

template <class TContainerDataIOTags>
struct ContainerDataIO {};

template <>
struct ContainerDataIO<ContainerDataIOTags::Historical>
{
static constexpr std::string_view mInfo = "Historical";

template <class TDataType>
static const TDataType& GetValue(
const ModelPart::NodeType& rNode,
const Variable<TDataType>& rVariable)
{
return rNode.FastGetSolutionStepValue(rVariable);
}

template <class TDataType>
static void SetValue(
ModelPart::NodeType& rNode,
const Variable<TDataType>& rVariable,
const TDataType& rValue)
{
rNode.FastGetSolutionStepValue(rVariable) = rValue;
}
};

template <>
struct ContainerDataIO<ContainerDataIOTags::NonHistorical>
{
static constexpr std::string_view mInfo = "NonHistorical";

template<class TDataType, class TEntityType>
static const TDataType& GetValue(
const TEntityType& rEntity,
const Variable<TDataType>& rVariable)
{
return rEntity.GetValue(rVariable);
}

template<class TDataType, class TEntityType>
static void SetValue(
TEntityType& rEntity,
const Variable<TDataType>& rVariable,
const TDataType& rValue)
{
rEntity.SetValue(rVariable, rValue);
}
};

} 