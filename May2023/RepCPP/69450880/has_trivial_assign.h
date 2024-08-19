




#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>

namespace hydra_thrust
{

namespace detail
{

template<typename T> struct has_trivial_assign
: public integral_constant<
bool,
(is_pod<T>::value && !is_const<T>::value)
#if HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_MSVC
|| __has_trivial_assign(T)
#elif HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_GCC
#if (__GNUC__ >= 4) && (__GNUC_MINOR__ >= 3)
|| __has_trivial_assign(T)
#endif 
#elif HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_CLANG
|| __has_trivial_assign(T)
#endif 
>
{};

} 

} 

