#pragma once

#include <algorithm> 
#include <array> 
#include <forward_list> 
#include <iterator> 
#include <map> 
#include <string> 
#include <tuple> 
#include <type_traits> 
#include <unordered_map> 
#include <utility> 
#include <valarray> 

#include <nlohmann/detail/boolean_operators.hpp>
#include <nlohmann/detail/exceptions.hpp>
#include <nlohmann/detail/macro_scope.hpp>
#include <nlohmann/detail/meta/cpp_future.hpp>
#include <nlohmann/detail/meta/type_traits.hpp>
#include <nlohmann/detail/value_t.hpp>

namespace nlohmann
{
namespace detail
{
template<typename BasicJsonType>
void from_json(const BasicJsonType& j, typename std::nullptr_t& n)
{
if (JSON_HEDLEY_UNLIKELY(not j.is_null()))
{
JSON_THROW(type_error::create(302, "type must be null, but is " + std::string(j.type_name())));
}
n = nullptr;
}

template<typename BasicJsonType, typename ArithmeticType,
enable_if_t<std::is_arithmetic<ArithmeticType>::value and
not std::is_same<ArithmeticType, typename BasicJsonType::boolean_t>::value,
int> = 0>
void get_arithmetic_value(const BasicJsonType& j, ArithmeticType& val)
{
switch (static_cast<value_t>(j))
{
case value_t::number_unsigned:
{
val = static_cast<ArithmeticType>(*j.template get_ptr<const typename BasicJsonType::number_unsigned_t*>());
break;
}
case value_t::number_integer:
{
val = static_cast<ArithmeticType>(*j.template get_ptr<const typename BasicJsonType::number_integer_t*>());
break;
}
case value_t::number_float:
{
val = static_cast<ArithmeticType>(*j.template get_ptr<const typename BasicJsonType::number_float_t*>());
break;
}

default:
JSON_THROW(type_error::create(302, "type must be number, but is " + std::string(j.type_name())));
}
}

template<typename BasicJsonType>
void from_json(const BasicJsonType& j, typename BasicJsonType::boolean_t& b)
{
if (JSON_HEDLEY_UNLIKELY(not j.is_boolean()))
{
JSON_THROW(type_error::create(302, "type must be boolean, but is " + std::string(j.type_name())));
}
b = *j.template get_ptr<const typename BasicJsonType::boolean_t*>();
}

template<typename BasicJsonType>
void from_json(const BasicJsonType& j, typename BasicJsonType::string_t& s)
{
if (JSON_HEDLEY_UNLIKELY(not j.is_string()))
{
JSON_THROW(type_error::create(302, "type must be string, but is " + std::string(j.type_name())));
}
s = *j.template get_ptr<const typename BasicJsonType::string_t*>();
}

template <
typename BasicJsonType, typename ConstructibleStringType,
enable_if_t <
is_constructible_string_type<BasicJsonType, ConstructibleStringType>::value and
not std::is_same<typename BasicJsonType::string_t,
ConstructibleStringType>::value,
int > = 0 >
void from_json(const BasicJsonType& j, ConstructibleStringType& s)
{
if (JSON_HEDLEY_UNLIKELY(not j.is_string()))
{
JSON_THROW(type_error::create(302, "type must be string, but is " + std::string(j.type_name())));
}

s = *j.template get_ptr<const typename BasicJsonType::string_t*>();
}

template<typename BasicJsonType>
void from_json(const BasicJsonType& j, typename BasicJsonType::number_float_t& val)
{
get_arithmetic_value(j, val);
}

template<typename BasicJsonType>
void from_json(const BasicJsonType& j, typename BasicJsonType::number_unsigned_t& val)
{
get_arithmetic_value(j, val);
}

template<typename BasicJsonType>
void from_json(const BasicJsonType& j, typename BasicJsonType::number_integer_t& val)
{
get_arithmetic_value(j, val);
}

template<typename BasicJsonType, typename EnumType,
enable_if_t<std::is_enum<EnumType>::value, int> = 0>
void from_json(const BasicJsonType& j, EnumType& e)
{
typename std::underlying_type<EnumType>::type val;
get_arithmetic_value(j, val);
e = static_cast<EnumType>(val);
}

template<typename BasicJsonType, typename T, typename Allocator,
enable_if_t<std::is_convertible<BasicJsonType, T>::value, int> = 0>
void from_json(const BasicJsonType& j, std::forward_list<T, Allocator>& l)
{
if (JSON_HEDLEY_UNLIKELY(not j.is_array()))
{
JSON_THROW(type_error::create(302, "type must be array, but is " + std::string(j.type_name())));
}
l.clear();
std::transform(j.rbegin(), j.rend(),
std::front_inserter(l), [](const BasicJsonType & i)
{
return i.template get<T>();
});
}

template<typename BasicJsonType, typename T,
enable_if_t<std::is_convertible<BasicJsonType, T>::value, int> = 0>
void from_json(const BasicJsonType& j, std::valarray<T>& l)
{
if (JSON_HEDLEY_UNLIKELY(not j.is_array()))
{
JSON_THROW(type_error::create(302, "type must be array, but is " + std::string(j.type_name())));
}
l.resize(j.size());
std::copy(j.begin(), j.end(), std::begin(l));
}

template <typename BasicJsonType, typename T, std::size_t N>
auto from_json(const BasicJsonType& j, T (&arr)[N])
-> decltype(j.template get<T>(), void())
{
for (std::size_t i = 0; i < N; ++i)
{
arr[i] = j.at(i).template get<T>();
}
}

template<typename BasicJsonType>
void from_json_array_impl(const BasicJsonType& j, typename BasicJsonType::array_t& arr, priority_tag<3> )
{
arr = *j.template get_ptr<const typename BasicJsonType::array_t*>();
}

template <typename BasicJsonType, typename T, std::size_t N>
auto from_json_array_impl(const BasicJsonType& j, std::array<T, N>& arr,
priority_tag<2> )
-> decltype(j.template get<T>(), void())
{
for (std::size_t i = 0; i < N; ++i)
{
arr[i] = j.at(i).template get<T>();
}
}

template<typename BasicJsonType, typename ConstructibleArrayType>
auto from_json_array_impl(const BasicJsonType& j, ConstructibleArrayType& arr, priority_tag<1> )
-> decltype(
arr.reserve(std::declval<typename ConstructibleArrayType::size_type>()),
j.template get<typename ConstructibleArrayType::value_type>(),
void())
{
using std::end;

ConstructibleArrayType ret;
ret.reserve(j.size());
std::transform(j.begin(), j.end(),
std::inserter(ret, end(ret)), [](const BasicJsonType & i)
{
return i.template get<typename ConstructibleArrayType::value_type>();
});
arr = std::move(ret);
}

template <typename BasicJsonType, typename ConstructibleArrayType>
void from_json_array_impl(const BasicJsonType& j, ConstructibleArrayType& arr,
priority_tag<0> )
{
using std::end;

ConstructibleArrayType ret;
std::transform(
j.begin(), j.end(), std::inserter(ret, end(ret)),
[](const BasicJsonType & i)
{
return i.template get<typename ConstructibleArrayType::value_type>();
});
arr = std::move(ret);
}

template <typename BasicJsonType, typename ConstructibleArrayType,
enable_if_t <
is_constructible_array_type<BasicJsonType, ConstructibleArrayType>::value and
not is_constructible_object_type<BasicJsonType, ConstructibleArrayType>::value and
not is_constructible_string_type<BasicJsonType, ConstructibleArrayType>::value and
not std::is_same<ConstructibleArrayType, typename BasicJsonType::binary_t>::value and
not is_basic_json<ConstructibleArrayType>::value,
int > = 0 >
auto from_json(const BasicJsonType& j, ConstructibleArrayType& arr)
-> decltype(from_json_array_impl(j, arr, priority_tag<3> {}),
j.template get<typename ConstructibleArrayType::value_type>(),
void())
{
if (JSON_HEDLEY_UNLIKELY(not j.is_array()))
{
JSON_THROW(type_error::create(302, "type must be array, but is " +
std::string(j.type_name())));
}

from_json_array_impl(j, arr, priority_tag<3> {});
}

template <typename BasicJsonType>
void from_json(const BasicJsonType& j, typename BasicJsonType::binary_t& bin)
{
if (JSON_HEDLEY_UNLIKELY(not j.is_binary()))
{
JSON_THROW(type_error::create(302, "type must be binary, but is " + std::string(j.type_name())));
}

bin = *j.template get_ptr<const typename BasicJsonType::binary_t*>();
}

template<typename BasicJsonType, typename ConstructibleObjectType,
enable_if_t<is_constructible_object_type<BasicJsonType, ConstructibleObjectType>::value, int> = 0>
void from_json(const BasicJsonType& j, ConstructibleObjectType& obj)
{
if (JSON_HEDLEY_UNLIKELY(not j.is_object()))
{
JSON_THROW(type_error::create(302, "type must be object, but is " + std::string(j.type_name())));
}

ConstructibleObjectType ret;
auto inner_object = j.template get_ptr<const typename BasicJsonType::object_t*>();
using value_type = typename ConstructibleObjectType::value_type;
std::transform(
inner_object->begin(), inner_object->end(),
std::inserter(ret, ret.begin()),
[](typename BasicJsonType::object_t::value_type const & p)
{
return value_type(p.first, p.second.template get<typename ConstructibleObjectType::mapped_type>());
});
obj = std::move(ret);
}

template<typename BasicJsonType, typename ArithmeticType,
enable_if_t <
std::is_arithmetic<ArithmeticType>::value and
not std::is_same<ArithmeticType, typename BasicJsonType::number_unsigned_t>::value and
not std::is_same<ArithmeticType, typename BasicJsonType::number_integer_t>::value and
not std::is_same<ArithmeticType, typename BasicJsonType::number_float_t>::value and
not std::is_same<ArithmeticType, typename BasicJsonType::boolean_t>::value,
int> = 0>
void from_json(const BasicJsonType& j, ArithmeticType& val)
{
switch (static_cast<value_t>(j))
{
case value_t::number_unsigned:
{
val = static_cast<ArithmeticType>(*j.template get_ptr<const typename BasicJsonType::number_unsigned_t*>());
break;
}
case value_t::number_integer:
{
val = static_cast<ArithmeticType>(*j.template get_ptr<const typename BasicJsonType::number_integer_t*>());
break;
}
case value_t::number_float:
{
val = static_cast<ArithmeticType>(*j.template get_ptr<const typename BasicJsonType::number_float_t*>());
break;
}
case value_t::boolean:
{
val = static_cast<ArithmeticType>(*j.template get_ptr<const typename BasicJsonType::boolean_t*>());
break;
}

default:
JSON_THROW(type_error::create(302, "type must be number, but is " + std::string(j.type_name())));
}
}

template<typename BasicJsonType, typename A1, typename A2>
void from_json(const BasicJsonType& j, std::pair<A1, A2>& p)
{
p = {j.at(0).template get<A1>(), j.at(1).template get<A2>()};
}

template<typename BasicJsonType, typename Tuple, std::size_t... Idx>
void from_json_tuple_impl(const BasicJsonType& j, Tuple& t, index_sequence<Idx...> )
{
t = std::make_tuple(j.at(Idx).template get<typename std::tuple_element<Idx, Tuple>::type>()...);
}

template<typename BasicJsonType, typename... Args>
void from_json(const BasicJsonType& j, std::tuple<Args...>& t)
{
from_json_tuple_impl(j, t, index_sequence_for<Args...> {});
}

template <typename BasicJsonType, typename Key, typename Value, typename Compare, typename Allocator,
typename = enable_if_t<not std::is_constructible<
typename BasicJsonType::string_t, Key>::value>>
void from_json(const BasicJsonType& j, std::map<Key, Value, Compare, Allocator>& m)
{
if (JSON_HEDLEY_UNLIKELY(not j.is_array()))
{
JSON_THROW(type_error::create(302, "type must be array, but is " + std::string(j.type_name())));
}
m.clear();
for (const auto& p : j)
{
if (JSON_HEDLEY_UNLIKELY(not p.is_array()))
{
JSON_THROW(type_error::create(302, "type must be array, but is " + std::string(p.type_name())));
}
m.emplace(p.at(0).template get<Key>(), p.at(1).template get<Value>());
}
}

template <typename BasicJsonType, typename Key, typename Value, typename Hash, typename KeyEqual, typename Allocator,
typename = enable_if_t<not std::is_constructible<
typename BasicJsonType::string_t, Key>::value>>
void from_json(const BasicJsonType& j, std::unordered_map<Key, Value, Hash, KeyEqual, Allocator>& m)
{
if (JSON_HEDLEY_UNLIKELY(not j.is_array()))
{
JSON_THROW(type_error::create(302, "type must be array, but is " + std::string(j.type_name())));
}
m.clear();
for (const auto& p : j)
{
if (JSON_HEDLEY_UNLIKELY(not p.is_array()))
{
JSON_THROW(type_error::create(302, "type must be array, but is " + std::string(p.type_name())));
}
m.emplace(p.at(0).template get<Key>(), p.at(1).template get<Value>());
}
}

struct from_json_fn
{
template<typename BasicJsonType, typename T>
auto operator()(const BasicJsonType& j, T& val) const
noexcept(noexcept(from_json(j, val)))
-> decltype(from_json(j, val), void())
{
return from_json(j, val);
}
};
}  

namespace
{
constexpr const auto& from_json = detail::static_const<detail::from_json_fn>::value;
} 
} 
