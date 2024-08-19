

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>

#define __HYDRA_THRUST_DEFINE_HAS_NESTED_TYPE(trait_name, nested_type_name) \
template<typename T> \
struct trait_name  \
{                    \
typedef char yes_type; \
typedef int  no_type;  \
template<typename S> static yes_type test(typename S::nested_type_name *); \
template<typename S> static no_type  test(...); \
static bool const value = sizeof(test<T>(0)) == sizeof(yes_type);\
typedef hydra_thrust::detail::integral_constant<bool, value> type;\
};

