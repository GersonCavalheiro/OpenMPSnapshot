#pragma once

#include <utility>

#include <nlohmann/detail/conversions/from_json.hpp>
#include <nlohmann/detail/conversions/to_json.hpp>

namespace nlohmann
{

template<typename, typename>
struct adl_serializer
{

template<typename BasicJsonType, typename ValueType>
static auto from_json(BasicJsonType&& j, ValueType& val) noexcept(
noexcept(::nlohmann::from_json(std::forward<BasicJsonType>(j), val)))
-> decltype(::nlohmann::from_json(std::forward<BasicJsonType>(j), val), void())
{
::nlohmann::from_json(std::forward<BasicJsonType>(j), val);
}


template <typename BasicJsonType, typename ValueType>
static auto to_json(BasicJsonType& j, ValueType&& val) noexcept(
noexcept(::nlohmann::to_json(j, std::forward<ValueType>(val))))
-> decltype(::nlohmann::to_json(j, std::forward<ValueType>(val)), void())
{
::nlohmann::to_json(j, std::forward<ValueType>(val));
}
};

}  
