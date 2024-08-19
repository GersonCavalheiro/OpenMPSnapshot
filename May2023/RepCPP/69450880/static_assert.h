



#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/preprocessor.h>

HYDRA_THRUST_BEGIN_NS

namespace detail
{

template <typename, bool x>
struct depend_on_instantiation
{
HYDRA_THRUST_INLINE_INTEGRAL_MEMBER_CONSTANT bool value = x;
};

#if HYDRA_THRUST_CPP_DIALECT >= 2011

#  if HYDRA_THRUST_CPP_DIALECT >= 2017
#    define HYDRA_THRUST_STATIC_ASSERT(B)        static_assert(B)
#  else
#    define HYDRA_THRUST_STATIC_ASSERT(B)        static_assert(B, "static assertion failed")
#  endif
#  define HYDRA_THRUST_STATIC_ASSERT_MSG(B, msg) static_assert(B, msg)

#else 

template <bool x> struct STATIC_ASSERTION_FAILURE;

template <> struct STATIC_ASSERTION_FAILURE<true> {};

template <int x> struct static_assert_test {};

#if    (  (HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_GCC)                  \
&& (HYDRA_THRUST_GCC_VERSION >= 40800))                                      \
|| (HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_CLANG)
#  define HYDRA_THRUST_STATIC_ASSERT(B)                                             \
typedef ::hydra_thrust::detail::static_assert_test<                             \
sizeof(::hydra_thrust::detail::STATIC_ASSERTION_FAILURE<(bool)(B)>)           \
>                                                                         \
HYDRA_THRUST_PP_CAT2(hydra_thrust_static_assert_typedef_, __LINE__)                 \
__attribute__((unused))                                                 \

#else
#  define HYDRA_THRUST_STATIC_ASSERT(B)                                             \
typedef ::hydra_thrust::detail::static_assert_test<                             \
sizeof(::hydra_thrust::detail::STATIC_ASSERTION_FAILURE<(bool)(B)>)           \
>                                                                         \
HYDRA_THRUST_PP_CAT2(hydra_thrust_static_assert_typedef_, __LINE__)                 \

#endif

#define HYDRA_THRUST_STATIC_ASSERT_MSG(B, msg) HYDRA_THRUST_STATIC_ASSERT(B)

#endif 

} 

HYDRA_THRUST_END_NS


