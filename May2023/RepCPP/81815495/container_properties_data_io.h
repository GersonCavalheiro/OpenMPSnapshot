
#pragma once


#include "containers/variable.h"
#include "containers/container_expression/container_data_io.h"
#include "includes/model_part.h"


namespace Kratos {


namespace ContainerDataIOTags {
struct Properties    {};
} 

template <>
struct ContainerDataIO<ContainerDataIOTags::Properties>
{
static constexpr std::string_view mInfo = "Properties";

template<class TDataType, class TEntityType>
static const TDataType& GetValue(
const TEntityType& rEntity,
const Variable<TDataType>& rVariable)
{
static_assert(!(std::is_same_v<TEntityType, ModelPart::NodeType>), "Properties retrieval is only supported for element and conditions.");
return rEntity.GetProperties().GetValue(rVariable);
}

template<class TDataType, class TEntityType>
static void SetValue(
TEntityType& rEntity,
const Variable<TDataType>& rVariable,
const TDataType& rValue)
{
static_assert(!(std::is_same_v<TEntityType, ModelPart::NodeType>), "Properties setter is only supported for element and conditions.");
rEntity.GetProperties().SetValue(rVariable, rValue);
}
};

} 