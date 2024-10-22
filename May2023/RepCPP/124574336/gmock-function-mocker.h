


#ifndef THIRD_PARTY_GOOGLETEST_GOOGLEMOCK_INCLUDE_GMOCK_INTERNAL_GMOCK_FUNCTION_MOCKER_H_  
#define THIRD_PARTY_GOOGLETEST_GOOGLEMOCK_INCLUDE_GMOCK_INTERNAL_GMOCK_FUNCTION_MOCKER_H_  

#include <type_traits>  
#include <utility>      

#include "gmock/gmock-spec-builders.h"
#include "gmock/internal/gmock-internal-utils.h"
#include "gmock/internal/gmock-pp.h"

namespace testing {
namespace internal {
template <typename T>
using identity_t = T;

template <typename MockType>
const MockType* AdjustConstness_const(const MockType* mock) {
return mock;
}

template <typename MockType>
MockType* AdjustConstness_(const MockType* mock) {
return const_cast<MockType*>(mock);
}

}  

using internal::FunctionMocker;
}  

#define MOCK_METHOD(...) \
GMOCK_PP_VARIADIC_CALL(GMOCK_INTERNAL_MOCK_METHOD_ARG_, __VA_ARGS__)

#define GMOCK_INTERNAL_MOCK_METHOD_ARG_1(...) \
GMOCK_INTERNAL_WRONG_ARITY(__VA_ARGS__)

#define GMOCK_INTERNAL_MOCK_METHOD_ARG_2(...) \
GMOCK_INTERNAL_WRONG_ARITY(__VA_ARGS__)

#define GMOCK_INTERNAL_MOCK_METHOD_ARG_3(_Ret, _MethodName, _Args) \
GMOCK_INTERNAL_MOCK_METHOD_ARG_4(_Ret, _MethodName, _Args, ())

#define GMOCK_INTERNAL_MOCK_METHOD_ARG_4(_Ret, _MethodName, _Args, _Spec)  \
GMOCK_INTERNAL_ASSERT_PARENTHESIS(_Args);                                \
GMOCK_INTERNAL_ASSERT_PARENTHESIS(_Spec);                                \
GMOCK_INTERNAL_ASSERT_VALID_SIGNATURE(                                   \
GMOCK_PP_NARG0 _Args, GMOCK_INTERNAL_SIGNATURE(_Ret, _Args));        \
GMOCK_INTERNAL_ASSERT_VALID_SPEC(_Spec)                                  \
GMOCK_INTERNAL_MOCK_METHOD_IMPL(                                         \
GMOCK_PP_NARG0 _Args, _MethodName, GMOCK_INTERNAL_HAS_CONST(_Spec),  \
GMOCK_INTERNAL_HAS_OVERRIDE(_Spec), GMOCK_INTERNAL_HAS_FINAL(_Spec), \
GMOCK_INTERNAL_GET_NOEXCEPT_SPEC(_Spec),                             \
GMOCK_INTERNAL_GET_CALLTYPE(_Spec),                                  \
(GMOCK_INTERNAL_SIGNATURE(_Ret, _Args)))

#define GMOCK_INTERNAL_MOCK_METHOD_ARG_5(...) \
GMOCK_INTERNAL_WRONG_ARITY(__VA_ARGS__)

#define GMOCK_INTERNAL_MOCK_METHOD_ARG_6(...) \
GMOCK_INTERNAL_WRONG_ARITY(__VA_ARGS__)

#define GMOCK_INTERNAL_MOCK_METHOD_ARG_7(...) \
GMOCK_INTERNAL_WRONG_ARITY(__VA_ARGS__)

#define GMOCK_INTERNAL_WRONG_ARITY(...)                                      \
static_assert(                                                             \
false,                                                                 \
"MOCK_METHOD must be called with 3 or 4 arguments. _Ret, "             \
"_MethodName, _Args and optionally _Spec. _Args and _Spec must be "    \
"enclosed in parentheses. If _Ret is a type with unprotected commas, " \
"it must also be enclosed in parentheses.")

#define GMOCK_INTERNAL_ASSERT_PARENTHESIS(_Tuple) \
static_assert(                                  \
GMOCK_PP_IS_ENCLOSED_PARENS(_Tuple),        \
GMOCK_PP_STRINGIZE(_Tuple) " should be enclosed in parentheses.")

#define GMOCK_INTERNAL_ASSERT_VALID_SIGNATURE(_N, ...)                 \
static_assert(                                                       \
std::is_function<__VA_ARGS__>::value,                            \
"Signature must be a function type, maybe return type contains " \
"unprotected comma.");                                           \
static_assert(                                                       \
::testing::tuple_size<typename ::testing::internal::Function<    \
__VA_ARGS__>::ArgumentTuple>::value == _N,               \
"This method does not take " GMOCK_PP_STRINGIZE(                 \
_N) " arguments. Parenthesize all types with unproctected commas.")

#define GMOCK_INTERNAL_ASSERT_VALID_SPEC(_Spec) \
GMOCK_PP_FOR_EACH(GMOCK_INTERNAL_ASSERT_VALID_SPEC_ELEMENT, ~, _Spec)

#define GMOCK_INTERNAL_MOCK_METHOD_IMPL(_N, _MethodName, _Constness,           \
_Override, _Final, _NoexceptSpec,      \
_CallType, _Signature)                 \
typename ::testing::internal::Function<GMOCK_PP_REMOVE_PARENS(               \
_Signature)>::Result                                                     \
GMOCK_INTERNAL_EXPAND(_CallType)                                             \
_MethodName(GMOCK_PP_REPEAT(GMOCK_INTERNAL_PARAMETER, _Signature, _N))   \
GMOCK_PP_IF(_Constness, const, ) _NoexceptSpec                       \
GMOCK_PP_IF(_Override, override, ) GMOCK_PP_IF(_Final, final, ) {    \
GMOCK_MOCKER_(_N, _Constness, _MethodName)                                 \
.SetOwnerAndName(this, #_MethodName);                                  \
return GMOCK_MOCKER_(_N, _Constness, _MethodName)                          \
.Invoke(GMOCK_PP_REPEAT(GMOCK_INTERNAL_FORWARD_ARG, _Signature, _N));  \
}                                                                            \
::testing::MockSpec<GMOCK_PP_REMOVE_PARENS(_Signature)> gmock_##_MethodName( \
GMOCK_PP_REPEAT(GMOCK_INTERNAL_MATCHER_PARAMETER, _Signature, _N))       \
GMOCK_PP_IF(_Constness, const, ) {                                       \
GMOCK_MOCKER_(_N, _Constness, _MethodName).RegisterOwner(this);            \
return GMOCK_MOCKER_(_N, _Constness, _MethodName)                          \
.With(GMOCK_PP_REPEAT(GMOCK_INTERNAL_MATCHER_ARGUMENT, , _N));         \
}                                                                            \
::testing::MockSpec<GMOCK_PP_REMOVE_PARENS(_Signature)> gmock_##_MethodName( \
const ::testing::internal::WithoutMatchers&,                             \
GMOCK_PP_IF(_Constness, const, )::testing::internal::Function<           \
GMOCK_PP_REMOVE_PARENS(_Signature)>*) const _NoexceptSpec {          \
return GMOCK_PP_CAT(::testing::internal::AdjustConstness_,                 \
GMOCK_PP_IF(_Constness, const, ))(this)                \
->gmock_##_MethodName(GMOCK_PP_REPEAT(                                 \
GMOCK_INTERNAL_A_MATCHER_ARGUMENT, _Signature, _N));               \
}                                                                            \
mutable ::testing::FunctionMocker<GMOCK_PP_REMOVE_PARENS(_Signature)>        \
GMOCK_MOCKER_(_N, _Constness, _MethodName)

#define GMOCK_INTERNAL_EXPAND(...) __VA_ARGS__

#define GMOCK_INTERNAL_HAS_CONST(_Tuple) \
GMOCK_PP_HAS_COMMA(GMOCK_PP_FOR_EACH(GMOCK_INTERNAL_DETECT_CONST, ~, _Tuple))

#define GMOCK_INTERNAL_HAS_OVERRIDE(_Tuple) \
GMOCK_PP_HAS_COMMA(                       \
GMOCK_PP_FOR_EACH(GMOCK_INTERNAL_DETECT_OVERRIDE, ~, _Tuple))

#define GMOCK_INTERNAL_HAS_FINAL(_Tuple) \
GMOCK_PP_HAS_COMMA(GMOCK_PP_FOR_EACH(GMOCK_INTERNAL_DETECT_FINAL, ~, _Tuple))

#define GMOCK_INTERNAL_GET_NOEXCEPT_SPEC(_Tuple) \
GMOCK_PP_FOR_EACH(GMOCK_INTERNAL_NOEXCEPT_SPEC_IF_NOEXCEPT, ~, _Tuple)

#define GMOCK_INTERNAL_NOEXCEPT_SPEC_IF_NOEXCEPT(_i, _, _elem)          \
GMOCK_PP_IF(                                                          \
GMOCK_PP_HAS_COMMA(GMOCK_INTERNAL_DETECT_NOEXCEPT(_i, _, _elem)), \
_elem, )

#define GMOCK_INTERNAL_GET_CALLTYPE(_Tuple) \
GMOCK_PP_FOR_EACH(GMOCK_INTERNAL_GET_CALLTYPE_IMPL, ~, _Tuple)

#define GMOCK_INTERNAL_ASSERT_VALID_SPEC_ELEMENT(_i, _, _elem)            \
static_assert(                                                          \
(GMOCK_PP_HAS_COMMA(GMOCK_INTERNAL_DETECT_CONST(_i, _, _elem)) +    \
GMOCK_PP_HAS_COMMA(GMOCK_INTERNAL_DETECT_OVERRIDE(_i, _, _elem)) + \
GMOCK_PP_HAS_COMMA(GMOCK_INTERNAL_DETECT_FINAL(_i, _, _elem)) +    \
GMOCK_PP_HAS_COMMA(GMOCK_INTERNAL_DETECT_NOEXCEPT(_i, _, _elem)) + \
GMOCK_INTERNAL_IS_CALLTYPE(_elem)) == 1,                           \
GMOCK_PP_STRINGIZE(                                                 \
_elem) " cannot be recognized as a valid specification modifier.");

#define GMOCK_INTERNAL_DETECT_CONST(_i, _, _elem) \
GMOCK_PP_CAT(GMOCK_INTERNAL_DETECT_CONST_I_, _elem)

#define GMOCK_INTERNAL_DETECT_CONST_I_const ,

#define GMOCK_INTERNAL_DETECT_OVERRIDE(_i, _, _elem) \
GMOCK_PP_CAT(GMOCK_INTERNAL_DETECT_OVERRIDE_I_, _elem)

#define GMOCK_INTERNAL_DETECT_OVERRIDE_I_override ,

#define GMOCK_INTERNAL_DETECT_FINAL(_i, _, _elem) \
GMOCK_PP_CAT(GMOCK_INTERNAL_DETECT_FINAL_I_, _elem)

#define GMOCK_INTERNAL_DETECT_FINAL_I_final ,

#define GMOCK_INTERNAL_DETECT_NOEXCEPT(_i, _, _elem) \
GMOCK_PP_CAT(GMOCK_INTERNAL_DETECT_NOEXCEPT_I_, _elem)

#define GMOCK_INTERNAL_DETECT_NOEXCEPT_I_noexcept ,

#define GMOCK_INTERNAL_GET_CALLTYPE_IMPL(_i, _, _elem)           \
GMOCK_PP_IF(GMOCK_INTERNAL_IS_CALLTYPE(_elem),                 \
GMOCK_INTERNAL_GET_VALUE_CALLTYPE, GMOCK_PP_EMPTY) \
(_elem)

#define GMOCK_INTERNAL_IS_CALLTYPE(_arg) \
GMOCK_INTERNAL_IS_CALLTYPE_I(          \
GMOCK_PP_CAT(GMOCK_INTERNAL_IS_CALLTYPE_HELPER_, _arg))
#define GMOCK_INTERNAL_IS_CALLTYPE_I(_arg) GMOCK_PP_IS_ENCLOSED_PARENS(_arg)

#define GMOCK_INTERNAL_GET_VALUE_CALLTYPE(_arg) \
GMOCK_INTERNAL_GET_VALUE_CALLTYPE_I(          \
GMOCK_PP_CAT(GMOCK_INTERNAL_IS_CALLTYPE_HELPER_, _arg))
#define GMOCK_INTERNAL_GET_VALUE_CALLTYPE_I(_arg) \
GMOCK_PP_IDENTITY _arg

#define GMOCK_INTERNAL_IS_CALLTYPE_HELPER_Calltype

#define GMOCK_INTERNAL_SIGNATURE(_Ret, _Args)                                 \
::testing::internal::identity_t<GMOCK_PP_IF(GMOCK_PP_IS_BEGIN_PARENS(_Ret), \
GMOCK_PP_REMOVE_PARENS,         \
GMOCK_PP_IDENTITY)(_Ret)>(      \
GMOCK_PP_FOR_EACH(GMOCK_INTERNAL_GET_TYPE, _, _Args))

#define GMOCK_INTERNAL_GET_TYPE(_i, _, _elem)                          \
GMOCK_PP_COMMA_IF(_i)                                                \
GMOCK_PP_IF(GMOCK_PP_IS_BEGIN_PARENS(_elem), GMOCK_PP_REMOVE_PARENS, \
GMOCK_PP_IDENTITY)                                       \
(_elem)

#define GMOCK_INTERNAL_PARAMETER(_i, _Signature, _)            \
GMOCK_PP_COMMA_IF(_i)                                        \
GMOCK_INTERNAL_ARG_O(_i, GMOCK_PP_REMOVE_PARENS(_Signature)) \
gmock_a##_i

#define GMOCK_INTERNAL_FORWARD_ARG(_i, _Signature, _) \
GMOCK_PP_COMMA_IF(_i)                               \
::std::forward<GMOCK_INTERNAL_ARG_O(                \
_i, GMOCK_PP_REMOVE_PARENS(_Signature))>(gmock_a##_i)

#define GMOCK_INTERNAL_MATCHER_PARAMETER(_i, _Signature, _)        \
GMOCK_PP_COMMA_IF(_i)                                            \
GMOCK_INTERNAL_MATCHER_O(_i, GMOCK_PP_REMOVE_PARENS(_Signature)) \
gmock_a##_i

#define GMOCK_INTERNAL_MATCHER_ARGUMENT(_i, _1, _2) \
GMOCK_PP_COMMA_IF(_i)                             \
gmock_a##_i

#define GMOCK_INTERNAL_A_MATCHER_ARGUMENT(_i, _Signature, _) \
GMOCK_PP_COMMA_IF(_i)                                      \
::testing::A<GMOCK_INTERNAL_ARG_O(_i, GMOCK_PP_REMOVE_PARENS(_Signature))>()

#define GMOCK_INTERNAL_ARG_O(_i, ...) \
typename ::testing::internal::Function<__VA_ARGS__>::template Arg<_i>::type

#define GMOCK_INTERNAL_MATCHER_O(_i, ...)                          \
const ::testing::Matcher<typename ::testing::internal::Function< \
__VA_ARGS__>::template Arg<_i>::type>&

#define MOCK_METHOD0(m, ...) GMOCK_INTERNAL_MOCK_METHODN(, , m, 0, __VA_ARGS__)
#define MOCK_METHOD1(m, ...) GMOCK_INTERNAL_MOCK_METHODN(, , m, 1, __VA_ARGS__)
#define MOCK_METHOD2(m, ...) GMOCK_INTERNAL_MOCK_METHODN(, , m, 2, __VA_ARGS__)
#define MOCK_METHOD3(m, ...) GMOCK_INTERNAL_MOCK_METHODN(, , m, 3, __VA_ARGS__)
#define MOCK_METHOD4(m, ...) GMOCK_INTERNAL_MOCK_METHODN(, , m, 4, __VA_ARGS__)
#define MOCK_METHOD5(m, ...) GMOCK_INTERNAL_MOCK_METHODN(, , m, 5, __VA_ARGS__)
#define MOCK_METHOD6(m, ...) GMOCK_INTERNAL_MOCK_METHODN(, , m, 6, __VA_ARGS__)
#define MOCK_METHOD7(m, ...) GMOCK_INTERNAL_MOCK_METHODN(, , m, 7, __VA_ARGS__)
#define MOCK_METHOD8(m, ...) GMOCK_INTERNAL_MOCK_METHODN(, , m, 8, __VA_ARGS__)
#define MOCK_METHOD9(m, ...) GMOCK_INTERNAL_MOCK_METHODN(, , m, 9, __VA_ARGS__)
#define MOCK_METHOD10(m, ...) \
GMOCK_INTERNAL_MOCK_METHODN(, , m, 10, __VA_ARGS__)

#define MOCK_CONST_METHOD0(m, ...) \
GMOCK_INTERNAL_MOCK_METHODN(const, , m, 0, __VA_ARGS__)
#define MOCK_CONST_METHOD1(m, ...) \
GMOCK_INTERNAL_MOCK_METHODN(const, , m, 1, __VA_ARGS__)
#define MOCK_CONST_METHOD2(m, ...) \
GMOCK_INTERNAL_MOCK_METHODN(const, , m, 2, __VA_ARGS__)
#define MOCK_CONST_METHOD3(m, ...) \
GMOCK_INTERNAL_MOCK_METHODN(const, , m, 3, __VA_ARGS__)
#define MOCK_CONST_METHOD4(m, ...) \
GMOCK_INTERNAL_MOCK_METHODN(const, , m, 4, __VA_ARGS__)
#define MOCK_CONST_METHOD5(m, ...) \
GMOCK_INTERNAL_MOCK_METHODN(const, , m, 5, __VA_ARGS__)
#define MOCK_CONST_METHOD6(m, ...) \
GMOCK_INTERNAL_MOCK_METHODN(const, , m, 6, __VA_ARGS__)
#define MOCK_CONST_METHOD7(m, ...) \
GMOCK_INTERNAL_MOCK_METHODN(const, , m, 7, __VA_ARGS__)
#define MOCK_CONST_METHOD8(m, ...) \
GMOCK_INTERNAL_MOCK_METHODN(const, , m, 8, __VA_ARGS__)
#define MOCK_CONST_METHOD9(m, ...) \
GMOCK_INTERNAL_MOCK_METHODN(const, , m, 9, __VA_ARGS__)
#define MOCK_CONST_METHOD10(m, ...) \
GMOCK_INTERNAL_MOCK_METHODN(const, , m, 10, __VA_ARGS__)

#define MOCK_METHOD0_T(m, ...) MOCK_METHOD0(m, __VA_ARGS__)
#define MOCK_METHOD1_T(m, ...) MOCK_METHOD1(m, __VA_ARGS__)
#define MOCK_METHOD2_T(m, ...) MOCK_METHOD2(m, __VA_ARGS__)
#define MOCK_METHOD3_T(m, ...) MOCK_METHOD3(m, __VA_ARGS__)
#define MOCK_METHOD4_T(m, ...) MOCK_METHOD4(m, __VA_ARGS__)
#define MOCK_METHOD5_T(m, ...) MOCK_METHOD5(m, __VA_ARGS__)
#define MOCK_METHOD6_T(m, ...) MOCK_METHOD6(m, __VA_ARGS__)
#define MOCK_METHOD7_T(m, ...) MOCK_METHOD7(m, __VA_ARGS__)
#define MOCK_METHOD8_T(m, ...) MOCK_METHOD8(m, __VA_ARGS__)
#define MOCK_METHOD9_T(m, ...) MOCK_METHOD9(m, __VA_ARGS__)
#define MOCK_METHOD10_T(m, ...) MOCK_METHOD10(m, __VA_ARGS__)

#define MOCK_CONST_METHOD0_T(m, ...) MOCK_CONST_METHOD0(m, __VA_ARGS__)
#define MOCK_CONST_METHOD1_T(m, ...) MOCK_CONST_METHOD1(m, __VA_ARGS__)
#define MOCK_CONST_METHOD2_T(m, ...) MOCK_CONST_METHOD2(m, __VA_ARGS__)
#define MOCK_CONST_METHOD3_T(m, ...) MOCK_CONST_METHOD3(m, __VA_ARGS__)
#define MOCK_CONST_METHOD4_T(m, ...) MOCK_CONST_METHOD4(m, __VA_ARGS__)
#define MOCK_CONST_METHOD5_T(m, ...) MOCK_CONST_METHOD5(m, __VA_ARGS__)
#define MOCK_CONST_METHOD6_T(m, ...) MOCK_CONST_METHOD6(m, __VA_ARGS__)
#define MOCK_CONST_METHOD7_T(m, ...) MOCK_CONST_METHOD7(m, __VA_ARGS__)
#define MOCK_CONST_METHOD8_T(m, ...) MOCK_CONST_METHOD8(m, __VA_ARGS__)
#define MOCK_CONST_METHOD9_T(m, ...) MOCK_CONST_METHOD9(m, __VA_ARGS__)
#define MOCK_CONST_METHOD10_T(m, ...) MOCK_CONST_METHOD10(m, __VA_ARGS__)

#define MOCK_METHOD0_WITH_CALLTYPE(ct, m, ...) \
GMOCK_INTERNAL_MOCK_METHODN(, ct, m, 0, __VA_ARGS__)
#define MOCK_METHOD1_WITH_CALLTYPE(ct, m, ...) \
GMOCK_INTERNAL_MOCK_METHODN(, ct, m, 1, __VA_ARGS__)
#define MOCK_METHOD2_WITH_CALLTYPE(ct, m, ...) \
GMOCK_INTERNAL_MOCK_METHODN(, ct, m, 2, __VA_ARGS__)
#define MOCK_METHOD3_WITH_CALLTYPE(ct, m, ...) \
GMOCK_INTERNAL_MOCK_METHODN(, ct, m, 3, __VA_ARGS__)
#define MOCK_METHOD4_WITH_CALLTYPE(ct, m, ...) \
GMOCK_INTERNAL_MOCK_METHODN(, ct, m, 4, __VA_ARGS__)
#define MOCK_METHOD5_WITH_CALLTYPE(ct, m, ...) \
GMOCK_INTERNAL_MOCK_METHODN(, ct, m, 5, __VA_ARGS__)
#define MOCK_METHOD6_WITH_CALLTYPE(ct, m, ...) \
GMOCK_INTERNAL_MOCK_METHODN(, ct, m, 6, __VA_ARGS__)
#define MOCK_METHOD7_WITH_CALLTYPE(ct, m, ...) \
GMOCK_INTERNAL_MOCK_METHODN(, ct, m, 7, __VA_ARGS__)
#define MOCK_METHOD8_WITH_CALLTYPE(ct, m, ...) \
GMOCK_INTERNAL_MOCK_METHODN(, ct, m, 8, __VA_ARGS__)
#define MOCK_METHOD9_WITH_CALLTYPE(ct, m, ...) \
GMOCK_INTERNAL_MOCK_METHODN(, ct, m, 9, __VA_ARGS__)
#define MOCK_METHOD10_WITH_CALLTYPE(ct, m, ...) \
GMOCK_INTERNAL_MOCK_METHODN(, ct, m, 10, __VA_ARGS__)

#define MOCK_CONST_METHOD0_WITH_CALLTYPE(ct, m, ...) \
GMOCK_INTERNAL_MOCK_METHODN(const, ct, m, 0, __VA_ARGS__)
#define MOCK_CONST_METHOD1_WITH_CALLTYPE(ct, m, ...) \
GMOCK_INTERNAL_MOCK_METHODN(const, ct, m, 1, __VA_ARGS__)
#define MOCK_CONST_METHOD2_WITH_CALLTYPE(ct, m, ...) \
GMOCK_INTERNAL_MOCK_METHODN(const, ct, m, 2, __VA_ARGS__)
#define MOCK_CONST_METHOD3_WITH_CALLTYPE(ct, m, ...) \
GMOCK_INTERNAL_MOCK_METHODN(const, ct, m, 3, __VA_ARGS__)
#define MOCK_CONST_METHOD4_WITH_CALLTYPE(ct, m, ...) \
GMOCK_INTERNAL_MOCK_METHODN(const, ct, m, 4, __VA_ARGS__)
#define MOCK_CONST_METHOD5_WITH_CALLTYPE(ct, m, ...) \
GMOCK_INTERNAL_MOCK_METHODN(const, ct, m, 5, __VA_ARGS__)
#define MOCK_CONST_METHOD6_WITH_CALLTYPE(ct, m, ...) \
GMOCK_INTERNAL_MOCK_METHODN(const, ct, m, 6, __VA_ARGS__)
#define MOCK_CONST_METHOD7_WITH_CALLTYPE(ct, m, ...) \
GMOCK_INTERNAL_MOCK_METHODN(const, ct, m, 7, __VA_ARGS__)
#define MOCK_CONST_METHOD8_WITH_CALLTYPE(ct, m, ...) \
GMOCK_INTERNAL_MOCK_METHODN(const, ct, m, 8, __VA_ARGS__)
#define MOCK_CONST_METHOD9_WITH_CALLTYPE(ct, m, ...) \
GMOCK_INTERNAL_MOCK_METHODN(const, ct, m, 9, __VA_ARGS__)
#define MOCK_CONST_METHOD10_WITH_CALLTYPE(ct, m, ...) \
GMOCK_INTERNAL_MOCK_METHODN(const, ct, m, 10, __VA_ARGS__)

#define MOCK_METHOD0_T_WITH_CALLTYPE(ct, m, ...) \
MOCK_METHOD0_WITH_CALLTYPE(ct, m, __VA_ARGS__)
#define MOCK_METHOD1_T_WITH_CALLTYPE(ct, m, ...) \
MOCK_METHOD1_WITH_CALLTYPE(ct, m, __VA_ARGS__)
#define MOCK_METHOD2_T_WITH_CALLTYPE(ct, m, ...) \
MOCK_METHOD2_WITH_CALLTYPE(ct, m, __VA_ARGS__)
#define MOCK_METHOD3_T_WITH_CALLTYPE(ct, m, ...) \
MOCK_METHOD3_WITH_CALLTYPE(ct, m, __VA_ARGS__)
#define MOCK_METHOD4_T_WITH_CALLTYPE(ct, m, ...) \
MOCK_METHOD4_WITH_CALLTYPE(ct, m, __VA_ARGS__)
#define MOCK_METHOD5_T_WITH_CALLTYPE(ct, m, ...) \
MOCK_METHOD5_WITH_CALLTYPE(ct, m, __VA_ARGS__)
#define MOCK_METHOD6_T_WITH_CALLTYPE(ct, m, ...) \
MOCK_METHOD6_WITH_CALLTYPE(ct, m, __VA_ARGS__)
#define MOCK_METHOD7_T_WITH_CALLTYPE(ct, m, ...) \
MOCK_METHOD7_WITH_CALLTYPE(ct, m, __VA_ARGS__)
#define MOCK_METHOD8_T_WITH_CALLTYPE(ct, m, ...) \
MOCK_METHOD8_WITH_CALLTYPE(ct, m, __VA_ARGS__)
#define MOCK_METHOD9_T_WITH_CALLTYPE(ct, m, ...) \
MOCK_METHOD9_WITH_CALLTYPE(ct, m, __VA_ARGS__)
#define MOCK_METHOD10_T_WITH_CALLTYPE(ct, m, ...) \
MOCK_METHOD10_WITH_CALLTYPE(ct, m, __VA_ARGS__)

#define MOCK_CONST_METHOD0_T_WITH_CALLTYPE(ct, m, ...) \
MOCK_CONST_METHOD0_WITH_CALLTYPE(ct, m, __VA_ARGS__)
#define MOCK_CONST_METHOD1_T_WITH_CALLTYPE(ct, m, ...) \
MOCK_CONST_METHOD1_WITH_CALLTYPE(ct, m, __VA_ARGS__)
#define MOCK_CONST_METHOD2_T_WITH_CALLTYPE(ct, m, ...) \
MOCK_CONST_METHOD2_WITH_CALLTYPE(ct, m, __VA_ARGS__)
#define MOCK_CONST_METHOD3_T_WITH_CALLTYPE(ct, m, ...) \
MOCK_CONST_METHOD3_WITH_CALLTYPE(ct, m, __VA_ARGS__)
#define MOCK_CONST_METHOD4_T_WITH_CALLTYPE(ct, m, ...) \
MOCK_CONST_METHOD4_WITH_CALLTYPE(ct, m, __VA_ARGS__)
#define MOCK_CONST_METHOD5_T_WITH_CALLTYPE(ct, m, ...) \
MOCK_CONST_METHOD5_WITH_CALLTYPE(ct, m, __VA_ARGS__)
#define MOCK_CONST_METHOD6_T_WITH_CALLTYPE(ct, m, ...) \
MOCK_CONST_METHOD6_WITH_CALLTYPE(ct, m, __VA_ARGS__)
#define MOCK_CONST_METHOD7_T_WITH_CALLTYPE(ct, m, ...) \
MOCK_CONST_METHOD7_WITH_CALLTYPE(ct, m, __VA_ARGS__)
#define MOCK_CONST_METHOD8_T_WITH_CALLTYPE(ct, m, ...) \
MOCK_CONST_METHOD8_WITH_CALLTYPE(ct, m, __VA_ARGS__)
#define MOCK_CONST_METHOD9_T_WITH_CALLTYPE(ct, m, ...) \
MOCK_CONST_METHOD9_WITH_CALLTYPE(ct, m, __VA_ARGS__)
#define MOCK_CONST_METHOD10_T_WITH_CALLTYPE(ct, m, ...) \
MOCK_CONST_METHOD10_WITH_CALLTYPE(ct, m, __VA_ARGS__)

#define GMOCK_INTERNAL_MOCK_METHODN(constness, ct, Method, args_num, ...) \
GMOCK_INTERNAL_ASSERT_VALID_SIGNATURE(                                  \
args_num, ::testing::internal::identity_t<__VA_ARGS__>);            \
GMOCK_INTERNAL_MOCK_METHOD_IMPL(                                        \
args_num, Method, GMOCK_PP_NARG0(constness), 0, 0, , ct,            \
(::testing::internal::identity_t<__VA_ARGS__>))

#define GMOCK_MOCKER_(arity, constness, Method) \
GTEST_CONCAT_TOKEN_(gmock##constness##arity##_##Method##_, __LINE__)

#endif  
