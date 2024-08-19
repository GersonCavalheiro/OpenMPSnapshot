

#pragma once

#include <cstddef>
#include <tuple>
#include <type_traits>

namespace thrill {
namespace common {

template <typename T>
struct FunctionTraits : public FunctionTraits<decltype(& T::operator ())>{ };

template <typename ClassType, typename ReturnType, typename ... Args>
struct FunctionTraits<ReturnType (ClassType::*)(Args ...) const>{

static constexpr size_t arity = sizeof ... (Args);

using result_type = ReturnType;
using is_const = std::true_type;

using args_tuple = std::tuple<Args ...>;

using args_tuple_plain = std::tuple<
typename std::remove_cv<
typename std::remove_reference<Args>::type>::type ...>;

template <size_t i>
using arg = typename std::tuple_element<i, args_tuple>::type;

template <size_t i>
using arg_plain =
typename std::remove_cv<
typename std::remove_reference<arg<i> >::type>::type;
};

template <typename ClassType, typename ReturnType, typename ... Args>
struct FunctionTraits<ReturnType (ClassType::*)(Args ...)>
: public FunctionTraits<ReturnType (ClassType::*)(Args ...) const>{
using is_const = std::false_type;
};

template <typename ReturnType, typename ... Args>
struct FunctionTraits<ReturnType (*)(Args ...)>{

static constexpr size_t arity = sizeof ... (Args);

using result_type = ReturnType;
using is_const = std::true_type;

using args_tuple = std::tuple<Args ...>;

using args_tuple_plain = std::tuple<
typename std::remove_cv<
typename std::remove_reference<Args>::type>::type ...>;

template <size_t i>
using arg = typename std::tuple_element<i, args_tuple>::type;

template <size_t i>
using arg_plain =
typename std::remove_cv<
typename std::remove_reference<arg<i> >::type>::type;
};

} 
} 