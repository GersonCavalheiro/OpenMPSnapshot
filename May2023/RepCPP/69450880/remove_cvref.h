

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>

HYDRA_THRUST_BEGIN_NS

#if HYDRA_THRUST_CPP_DIALECT >= 2020

using std::remove_cvref;
using std::remove_cvref_t;

#else 

template <typename T>
struct remove_cvref
{
typedef typename detail::remove_cv<
typename detail::remove_reference<T>::type
>::type type;
};

#if HYDRA_THRUST_CPP_DIALECT >= 2011
template <typename T>
using remove_cvref_t = typename remove_cvref<T>::type;
#endif

#endif 

HYDRA_THRUST_END_NS

