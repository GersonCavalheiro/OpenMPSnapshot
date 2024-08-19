#pragma once

#include <utility> 
#include <nlohmann/thirdparty/hedley/hedley.hpp>


#if !defined(JSON_SKIP_UNSUPPORTED_COMPILER_CHECK)
#if defined(__clang__)
#if (__clang_major__ * 10000 + __clang_minor__ * 100 + __clang_patchlevel__) < 30400
#error "unsupported Clang version - see https:
#endif
#elif defined(__GNUC__) && !(defined(__ICC) || defined(__INTEL_COMPILER))
#if (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) < 40800
#error "unsupported GCC version - see https:
#endif
#endif
#endif

#if (defined(__cplusplus) && __cplusplus >= 201703L) || (defined(_HAS_CXX17) && _HAS_CXX17 == 1) 
#define JSON_HAS_CPP_17
#define JSON_HAS_CPP_14
#elif (defined(__cplusplus) && __cplusplus >= 201402L) || (defined(_HAS_CXX14) && _HAS_CXX14 == 1)
#define JSON_HAS_CPP_14
#endif

#if defined(__clang__) || defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
#endif

#if defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdocumentation"
#endif

#if (defined(__cpp_exceptions) || defined(__EXCEPTIONS) || defined(_CPPUNWIND)) && !defined(JSON_NOEXCEPTION)
#define JSON_THROW(exception) throw exception
#define JSON_TRY try
#define JSON_CATCH(exception) catch(exception)
#define JSON_INTERNAL_CATCH(exception) catch(exception)
#else
#include <cstdlib>
#define JSON_THROW(exception) std::abort()
#define JSON_TRY if(true)
#define JSON_CATCH(exception) if(false)
#define JSON_INTERNAL_CATCH(exception) if(false)
#endif

#if defined(JSON_THROW_USER)
#undef JSON_THROW
#define JSON_THROW JSON_THROW_USER
#endif
#if defined(JSON_TRY_USER)
#undef JSON_TRY
#define JSON_TRY JSON_TRY_USER
#endif
#if defined(JSON_CATCH_USER)
#undef JSON_CATCH
#define JSON_CATCH JSON_CATCH_USER
#undef JSON_INTERNAL_CATCH
#define JSON_INTERNAL_CATCH JSON_CATCH_USER
#endif
#if defined(JSON_INTERNAL_CATCH_USER)
#undef JSON_INTERNAL_CATCH
#define JSON_INTERNAL_CATCH JSON_INTERNAL_CATCH_USER
#endif


#define NLOHMANN_JSON_SERIALIZE_ENUM(ENUM_TYPE, ...)                                            \
template<typename BasicJsonType>                                                            \
inline void to_json(BasicJsonType& j, const ENUM_TYPE& e)                                   \
{                                                                                           \
static_assert(std::is_enum<ENUM_TYPE>::value, #ENUM_TYPE " must be an enum!");          \
static const std::pair<ENUM_TYPE, BasicJsonType> m[] = __VA_ARGS__;                     \
auto it = std::find_if(std::begin(m), std::end(m),                                      \
[e](const std::pair<ENUM_TYPE, BasicJsonType>& ej_pair) -> bool  \
{                                                                                       \
return ej_pair.first == e;                                                          \
});                                                                                     \
j = ((it != std::end(m)) ? it : std::begin(m))->second;                                 \
}                                                                                           \
template<typename BasicJsonType>                                                            \
inline void from_json(const BasicJsonType& j, ENUM_TYPE& e)                                 \
{                                                                                           \
static_assert(std::is_enum<ENUM_TYPE>::value, #ENUM_TYPE " must be an enum!");          \
static const std::pair<ENUM_TYPE, BasicJsonType> m[] = __VA_ARGS__;                     \
auto it = std::find_if(std::begin(m), std::end(m),                                      \
[&j](const std::pair<ENUM_TYPE, BasicJsonType>& ej_pair) -> bool \
{                                                                                       \
return ej_pair.second == j;                                                         \
});                                                                                     \
e = ((it != std::end(m)) ? it : std::begin(m))->first;                                  \
}


#define NLOHMANN_BASIC_JSON_TPL_DECLARATION                                \
template<template<typename, typename, typename...> class ObjectType,   \
template<typename, typename...> class ArrayType,              \
class StringType, class BooleanType, class NumberIntegerType, \
class NumberUnsignedType, class NumberFloatType,              \
template<typename> class AllocatorType,                       \
template<typename, typename = void> class JSONSerializer,     \
class BinaryType>

#define NLOHMANN_BASIC_JSON_TPL                                            \
basic_json<ObjectType, ArrayType, StringType, BooleanType,             \
NumberIntegerType, NumberUnsignedType, NumberFloatType,                \
AllocatorType, JSONSerializer, BinaryType>
