

#ifndef BOOST_OUTCOME_TRY_HPP
#define BOOST_OUTCOME_TRY_HPP

#include "success_failure.hpp"

BOOST_OUTCOME_V2_NAMESPACE_BEGIN

namespace detail
{
struct has_value_overload
{
};
struct as_failure_overload
{
};
struct assume_error_overload
{
};
struct error_overload
{
};
struct assume_value_overload
{
};
struct value_overload
{
};
BOOST_OUTCOME_TEMPLATE(class T, class R = decltype(std::declval<T>().as_failure()))
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(BOOST_OUTCOME_V2_NAMESPACE::is_failure_type<R>))
constexpr inline bool has_as_failure(int ) { return true; }
template <class T> constexpr inline bool has_as_failure(...) { return false; }
BOOST_OUTCOME_TEMPLATE(class T)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TEXPR(std::declval<T>().assume_error()))
constexpr inline bool has_assume_error(int ) { return true; }
template <class T> constexpr inline bool has_assume_error(...) { return false; }
BOOST_OUTCOME_TEMPLATE(class T)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TEXPR(std::declval<T>().error()))
constexpr inline bool has_error(int ) { return true; }
template <class T> constexpr inline bool has_error(...) { return false; }
BOOST_OUTCOME_TEMPLATE(class T)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TEXPR(std::declval<T>().assume_value()))
constexpr inline bool has_assume_value(int ) { return true; }
template <class T> constexpr inline bool has_assume_value(...) { return false; }
BOOST_OUTCOME_TEMPLATE(class T)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TEXPR(std::declval<T>().value()))
constexpr inline bool has_value(int ) { return true; }
template <class T> constexpr inline bool has_value(...) { return false; }
}  


BOOST_OUTCOME_TEMPLATE(class T)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TEXPR(std::declval<T>().has_value()))
constexpr inline bool try_operation_has_value(T &&v, detail::has_value_overload = {})
{
return v.has_value();
}


BOOST_OUTCOME_TEMPLATE(class T)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(detail::has_as_failure<T>(5)))
constexpr inline decltype(auto) try_operation_return_as(T &&v, detail::as_failure_overload = {})
{
return static_cast<T &&>(v).as_failure();
}

BOOST_OUTCOME_TEMPLATE(class T)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(!detail::has_as_failure<T>(5) && detail::has_assume_error<T>(5)))
constexpr inline decltype(auto) try_operation_return_as(T &&v, detail::assume_error_overload = {})
{
return failure(static_cast<T &&>(v).assume_error());
}

BOOST_OUTCOME_TEMPLATE(class T)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(!detail::has_as_failure<T>(5) && !detail::has_assume_error<T>(5) && detail::has_error<T>(5)))
constexpr inline decltype(auto) try_operation_return_as(T &&v, detail::error_overload = {})
{
return failure(static_cast<T &&>(v).error());
}


BOOST_OUTCOME_TEMPLATE(class T)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(detail::has_assume_value<T>(5)))
constexpr inline decltype(auto) try_operation_extract_value(T &&v, detail::assume_value_overload = {})
{
return static_cast<T &&>(v).assume_value();
}

BOOST_OUTCOME_TEMPLATE(class T)
BOOST_OUTCOME_TREQUIRES(BOOST_OUTCOME_TPRED(!detail::has_assume_value<T>(5) && detail::has_value<T>(5)))
constexpr inline decltype(auto) try_operation_extract_value(T &&v, detail::value_overload = {})
{
return static_cast<T &&>(v).value();
}

BOOST_OUTCOME_V2_NAMESPACE_END

#if !defined(__clang__) && defined(__GNUC__) && __GNUC__ >= 8
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wparentheses"
#endif


#define BOOST_OUTCOME_TRY_GLUE2(x, y) x##y
#define BOOST_OUTCOME_TRY_GLUE(x, y) BOOST_OUTCOME_TRY_GLUE2(x, y)
#define BOOST_OUTCOME_TRY_UNIQUE_NAME BOOST_OUTCOME_TRY_GLUE(_outcome_try_unique_name_temporary, __COUNTER__)

#define BOOST_OUTCOME_TRY_RETURN_ARG_COUNT(_1_, _2_, _3_, _4_, _5_, _6_, _7_, _8_, count, ...) count
#define BOOST_OUTCOME_TRY_EXPAND_ARGS(args) BOOST_OUTCOME_TRY_RETURN_ARG_COUNT args
#define BOOST_OUTCOME_TRY_COUNT_ARGS_MAX8(...) BOOST_OUTCOME_TRY_EXPAND_ARGS((__VA_ARGS__, 8, 7, 6, 5, 4, 3, 2, 1, 0))
#define BOOST_OUTCOME_TRY_OVERLOAD_MACRO2(name, count) name##count
#define BOOST_OUTCOME_TRY_OVERLOAD_MACRO1(name, count) BOOST_OUTCOME_TRY_OVERLOAD_MACRO2(name, count)
#define BOOST_OUTCOME_TRY_OVERLOAD_MACRO(name, count) BOOST_OUTCOME_TRY_OVERLOAD_MACRO1(name, count)
#define BOOST_OUTCOME_TRY_OVERLOAD_GLUE(x, y) x y
#define BOOST_OUTCOME_TRY_CALL_OVERLOAD(name, ...)                                                                                                                   \
BOOST_OUTCOME_TRY_OVERLOAD_GLUE(BOOST_OUTCOME_TRY_OVERLOAD_MACRO(name, BOOST_OUTCOME_TRY_COUNT_ARGS_MAX8(__VA_ARGS__)), (__VA_ARGS__))

#ifndef BOOST_OUTCOME_TRY_LIKELY
#if defined(__clang__) || defined(__GNUC__)
#define BOOST_OUTCOME_TRY_LIKELY(expr) (__builtin_expect(!!(expr), true))
#else
#define BOOST_OUTCOME_TRY_LIKELY(expr) (expr)
#endif
#endif

#define BOOST_OUTCOME_TRYV2_SUCCESS_LIKELY(unique, ...)                                                                                                              \
auto &&unique = (__VA_ARGS__);                                                                                                                               \
if(BOOST_OUTCOME_TRY_LIKELY(BOOST_OUTCOME_V2_NAMESPACE::try_operation_has_value(unique)))                                                                                \
;                                                                                                                                                          \
else                                                                                                                                                         \
return BOOST_OUTCOME_V2_NAMESPACE::try_operation_return_as(static_cast<decltype(unique) &&>(unique))
#define BOOST_OUTCOME_TRY2_SUCCESS_LIKELY(unique, v, ...)                                                                                                            \
BOOST_OUTCOME_TRYV2_SUCCESS_LIKELY(unique, __VA_ARGS__);                                                                                                           \
v = BOOST_OUTCOME_V2_NAMESPACE::try_operation_extract_value(static_cast<decltype(unique) &&>(unique))
#define BOOST_OUTCOME_TRYV2_FAILURE_LIKELY(unique, ...)                                                                                                              \
auto &&unique = (__VA_ARGS__);                                                                                                                               \
if(BOOST_OUTCOME_TRY_LIKELY(!BOOST_OUTCOME_V2_NAMESPACE::try_operation_has_value(unique)))                                                                               \
return BOOST_OUTCOME_V2_NAMESPACE::try_operation_return_as(static_cast<decltype(unique) &&>(unique))
#define BOOST_OUTCOME_TRY2_FAILURE_LIKELY(unique, v, ...)                                                                                                            \
BOOST_OUTCOME_TRYV2_FAILURE_LIKELY(unique, __VA_ARGS__);                                                                                                           \
v = BOOST_OUTCOME_V2_NAMESPACE::try_operation_extract_value(static_cast<decltype(unique) &&>(unique))

#define BOOST_OUTCOME_CO_TRYV2_SUCCESS_LIKELY(unique, ...)                                                                                                           \
auto &&unique = (__VA_ARGS__);                                                                                                                               \
if(BOOST_OUTCOME_TRY_LIKELY(BOOST_OUTCOME_V2_NAMESPACE::try_operation_has_value(unique)))                                                                                \
;                                                                                                                                                          \
else                                                                                                                                                         \
co_return BOOST_OUTCOME_V2_NAMESPACE::try_operation_return_as(static_cast<decltype(unique) &&>(unique))
#define BOOST_OUTCOME_CO_TRY2_SUCCESS_LIKELY(unique, v, ...)                                                                                                         \
BOOST_OUTCOME_CO_TRYV2_SUCCESS_LIKELY(unique, __VA_ARGS__);                                                                                                        \
v = BOOST_OUTCOME_V2_NAMESPACE::try_operation_extract_value(static_cast<decltype(unique) &&>(unique))
#define BOOST_OUTCOME_CO_TRYV2_FAILURE_LIKELY(unique, ...)                                                                                                           \
auto &&unique = (__VA_ARGS__);                                                                                                                               \
if(BOOST_OUTCOME_TRY_LIKELY(!BOOST_OUTCOME_V2_NAMESPACE::try_operation_has_value(unique)))                                                                               \
co_return BOOST_OUTCOME_V2_NAMESPACE::try_operation_return_as(static_cast<decltype(unique) &&>(unique))
#define BOOST_OUTCOME_CO_TRY2_FAILURE_LIKELY(unique, v, ...)                                                                                                         \
BOOST_OUTCOME_CO_TRYV2_FAILURE_LIKELY(unique, __VA_ARGS__);                                                                                                        \
v = BOOST_OUTCOME_V2_NAMESPACE::try_operation_extract_value(static_cast<decltype(unique) &&>(unique))


#define BOOST_OUTCOME_TRYV(...) BOOST_OUTCOME_TRYV2_SUCCESS_LIKELY(BOOST_OUTCOME_TRY_UNIQUE_NAME, __VA_ARGS__)

#define BOOST_OUTCOME_TRYV_FAILURE_LIKELY(...) BOOST_OUTCOME_TRYV2_FAILURE_LIKELY(BOOST_OUTCOME_TRY_UNIQUE_NAME, __VA_ARGS__)


#define BOOST_OUTCOME_CO_TRYV(...) BOOST_OUTCOME_CO_TRYV2_SUCCESS_LIKELY(BOOST_OUTCOME_TRY_UNIQUE_NAME, __VA_ARGS__)

#define BOOST_OUTCOME_CO_TRYV_FAILURE_LIKELY(...) BOOST_OUTCOME_CO_TRYV2_FAILURE_LIKELY(BOOST_OUTCOME_TRY_UNIQUE_NAME, __VA_ARGS__)

#if defined(__GNUC__) || defined(__clang__)

#define BOOST_OUTCOME_TRYX2(unique, retstmt, ...)                                                                                                                    \
({                                                                                                                                                           \
auto &&unique = (__VA_ARGS__);                                                                                                                             \
if(BOOST_OUTCOME_TRY_LIKELY(BOOST_OUTCOME_V2_NAMESPACE::try_operation_has_value(unique)))                                                                              \
;                                                                                                                                                        \
else                                                                                                                                                       \
retstmt BOOST_OUTCOME_V2_NAMESPACE::try_operation_return_as(static_cast<decltype(unique) &&>(unique));                                                         \
BOOST_OUTCOME_V2_NAMESPACE::try_operation_extract_value(static_cast<decltype(unique) &&>(unique));                                                               \
})


#define BOOST_OUTCOME_TRYX(...) BOOST_OUTCOME_TRYX2(BOOST_OUTCOME_TRY_UNIQUE_NAME, return, __VA_ARGS__)


#define BOOST_OUTCOME_CO_TRYX(...) BOOST_OUTCOME_TRYX2(BOOST_OUTCOME_TRY_UNIQUE_NAME, co_return, __VA_ARGS__)
#endif


#define BOOST_OUTCOME_TRYA(v, ...) BOOST_OUTCOME_TRY2_SUCCESS_LIKELY(BOOST_OUTCOME_TRY_UNIQUE_NAME, auto &&v, __VA_ARGS__)

#define BOOST_OUTCOME_TRYA_FAILURE_LIKELY(v, ...) BOOST_OUTCOME_TRY2_FAILURE_LIKELY(BOOST_OUTCOME_TRY_UNIQUE_NAME, auto &&v, __VA_ARGS__)


#define BOOST_OUTCOME_CO_TRYA(v, ...) BOOST_OUTCOME_CO_TRY2_SUCCESS_LIKELY(BOOST_OUTCOME_TRY_UNIQUE_NAME, auto &&v, __VA_ARGS__)

#define BOOST_OUTCOME_CO_TRYA_FAILURE_LIKELY(v, ...) BOOST_OUTCOME_CO_TRY2_FAILURE_LIKELY(BOOST_OUTCOME_TRY_UNIQUE_NAME, auto &&v, __VA_ARGS__)


#define BOOST_OUTCOME_TRY_INVOKE_TRY8(a, b, c, d, e, f, g, h) BOOST_OUTCOME_TRYA(a, b, c, d, e, f, g, h)
#define BOOST_OUTCOME_TRY_INVOKE_TRY7(a, b, c, d, e, f, g) BOOST_OUTCOME_TRYA(a, b, c, d, e, f, g)
#define BOOST_OUTCOME_TRY_INVOKE_TRY6(a, b, c, d, e, f) BOOST_OUTCOME_TRYA(a, b, c, d, e, f)
#define BOOST_OUTCOME_TRY_INVOKE_TRY5(a, b, c, d, e) BOOST_OUTCOME_TRYA(a, b, c, d, e)
#define BOOST_OUTCOME_TRY_INVOKE_TRY4(a, b, c, d) BOOST_OUTCOME_TRYA(a, b, c, d)
#define BOOST_OUTCOME_TRY_INVOKE_TRY3(a, b, c) BOOST_OUTCOME_TRYA(a, b, c)
#define BOOST_OUTCOME_TRY_INVOKE_TRY2(a, b) BOOST_OUTCOME_TRYA(a, b)
#define BOOST_OUTCOME_TRY_INVOKE_TRY1(a) BOOST_OUTCOME_TRYV(a)


#define BOOST_OUTCOME_TRY(...) BOOST_OUTCOME_TRY_CALL_OVERLOAD(BOOST_OUTCOME_TRY_INVOKE_TRY, __VA_ARGS__)

#define BOOST_OUTCOME_TRY_FAILURE_LIKELY_INVOKE_TRY8(a, b, c, d, e, f, g, h) BOOST_OUTCOME_TRYA_FAILURE_LIKELY(a, b, c, d, e, f, g, h)
#define BOOST_OUTCOME_TRY_FAILURE_LIKELY_INVOKE_TRY7(a, b, c, d, e, f, g) BOOST_OUTCOME_TRYA_FAILURE_LIKELY(a, b, c, d, e, f, g)
#define BOOST_OUTCOME_TRY_FAILURE_LIKELY_INVOKE_TRY6(a, b, c, d, e, f) BOOST_OUTCOME_TRYA_FAILURE_LIKELY(a, b, c, d, e, f)
#define BOOST_OUTCOME_TRY_FAILURE_LIKELY_INVOKE_TRY5(a, b, c, d, e) BOOST_OUTCOME_TRYA_FAILURE_LIKELY(a, b, c, d, e)
#define BOOST_OUTCOME_TRY_FAILURE_LIKELY_INVOKE_TRY4(a, b, c, d) BOOST_OUTCOME_TRYA_FAILURE_LIKELY(a, b, c, d)
#define BOOST_OUTCOME_TRY_FAILURE_LIKELY_INVOKE_TRY3(a, b, c) BOOST_OUTCOME_TRYA_FAILURE_LIKELY(a, b, c)
#define BOOST_OUTCOME_TRY_FAILURE_LIKELY_INVOKE_TRY2(a, b) BOOST_OUTCOME_TRYA_FAILURE_LIKELY(a, b)
#define BOOST_OUTCOME_TRY_FAILURE_LIKELY_INVOKE_TRY1(a) BOOST_OUTCOME_TRYV_FAILURE_LIKELY(a)

#define BOOST_OUTCOME_TRY_FAILURE_LIKELY(...) BOOST_OUTCOME_TRY_CALL_OVERLOAD(BOOST_OUTCOME_TRY_FAILURE_LIKELY_INVOKE_TRY, __VA_ARGS__)

#define BOOST_OUTCOME_CO_TRY_INVOKE_TRY8(a, b, c, d, e, f, g, h) BOOST_OUTCOME_CO_TRYA(a, b, c, d, e, f, g, h)
#define BOOST_OUTCOME_CO_TRY_INVOKE_TRY7(a, b, c, d, e, f, g) BOOST_OUTCOME_CO_TRYA(a, b, c, d, e, f, g)
#define BOOST_OUTCOME_CO_TRY_INVOKE_TRY6(a, b, c, d, e, f) BOOST_OUTCOME_CO_TRYA(a, b, c, d, e, f)
#define BOOST_OUTCOME_CO_TRY_INVOKE_TRY5(a, b, c, d, e) BOOST_OUTCOME_CO_TRYA(a, b, c, d, e)
#define BOOST_OUTCOME_CO_TRY_INVOKE_TRY4(a, b, c, d) BOOST_OUTCOME_CO_TRYA(a, b, c, d)
#define BOOST_OUTCOME_CO_TRY_INVOKE_TRY3(a, b, c) BOOST_OUTCOME_CO_TRYA(a, b, c)
#define BOOST_OUTCOME_CO_TRY_INVOKE_TRY2(a, b) BOOST_OUTCOME_CO_TRYA(a, b)
#define BOOST_OUTCOME_CO_TRY_INVOKE_TRY1(a) BOOST_OUTCOME_CO_TRYV(a)

#define BOOST_OUTCOME_CO_TRY(...) BOOST_OUTCOME_TRY_CALL_OVERLOAD(BOOST_OUTCOME_CO_TRY_INVOKE_TRY, __VA_ARGS__)


#define BOOST_OUTCOME_TRY(...) BOOST_OUTCOME_TRY_CALL_OVERLOAD(BOOST_OUTCOME_TRY_INVOKE_TRY, __VA_ARGS__)

#define BOOST_OUTCOME_CO_TRY_FAILURE_LIKELY_INVOKE_TRY8(a, b, c, d, e, f, g, h) BOOST_OUTCOME_CO_TRYA_FAILURE_LIKELY(a, b, c, d, e, f, g, h)
#define BOOST_OUTCOME_CO_TRY_FAILURE_LIKELY_INVOKE_TRY7(a, b, c, d, e, f, g) BOOST_OUTCOME_CO_TRYA_FAILURE_LIKELY(a, b, c, d, e, f, g)
#define BOOST_OUTCOME_CO_TRY_FAILURE_LIKELY_INVOKE_TRY6(a, b, c, d, e, f) BOOST_OUTCOME_CO_TRYA_FAILURE_LIKELY(a, b, c, d, e, f)
#define BOOST_OUTCOME_CO_TRY_FAILURE_LIKELY_INVOKE_TRY5(a, b, c, d, e) BOOST_OUTCOME_CO_TRYA_FAILURE_LIKELY(a, b, c, d, e)
#define BOOST_OUTCOME_CO_TRY_FAILURE_LIKELY_INVOKE_TRY4(a, b, c, d) BOOST_OUTCOME_CO_TRYA_FAILURE_LIKELY(a, b, c, d)
#define BOOST_OUTCOME_CO_TRY_FAILURE_LIKELY_INVOKE_TRY3(a, b, c) BOOST_OUTCOME_CO_TRYA_FAILURE_LIKELY(a, b, c)
#define BOOST_OUTCOME_CO_TRY_FAILURE_LIKELY_INVOKE_TRY2(a, b) BOOST_OUTCOME_CO_TRYA_FAILURE_LIKELY(a, b)
#define BOOST_OUTCOME_CO_TRY_FAILURE_LIKELY_INVOKE_TRY1(a) BOOST_OUTCOME_CO_TRYV_FAILURE_LIKELY(a)

#define BOOST_OUTCOME_CO_TRY_FAILURE_LIKELY(...) BOOST_OUTCOME_TRY_CALL_OVERLOAD(BOOST_OUTCOME_CO_TRY_FAILURE_LIKELY_INVOKE_TRY, __VA_ARGS__)

#if !defined(__clang__) && defined(__GNUC__) && __GNUC__ >= 8
#pragma GCC diagnostic pop
#endif

#endif
