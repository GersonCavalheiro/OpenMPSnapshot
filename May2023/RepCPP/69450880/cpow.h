

#pragma once

#include <hydra/detail/external/hydra_thrust/complex.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>

namespace hydra_thrust {

template <typename T0, typename T1>
__host__ __device__
complex<typename detail::promoted_numerical_type<T0, T1>::type>
pow(const complex<T0>& x, const complex<T1>& y)
{
typedef typename detail::promoted_numerical_type<T0, T1>::type T;
return exp(log(complex<T>(x)) * complex<T>(y));
}

template <typename T0, typename T1>
__host__ __device__
complex<typename detail::promoted_numerical_type<T0, T1>::type>
pow(const complex<T0>& x, const T1& y)
{
typedef typename detail::promoted_numerical_type<T0, T1>::type T;
return exp(log(complex<T>(x)) * T(y));
}

template <typename T0, typename T1>
__host__ __device__
complex<typename detail::promoted_numerical_type<T0, T1>::type>
pow(const T0& x, const complex<T1>& y)
{
typedef typename detail::promoted_numerical_type<T0, T1>::type T;
using std::log;
return exp(log(T(x)) * complex<T>(y));
}

} 

