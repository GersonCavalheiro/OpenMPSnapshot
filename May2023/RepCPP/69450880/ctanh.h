







#pragma once

#include <hydra/detail/external/hydra_thrust/complex.h>
#include <hydra/detail/external/hydra_thrust/detail/complex/math_private.h>
#include <cmath>

namespace hydra_thrust{
namespace detail{
namespace complex{		      	

using hydra_thrust::complex;

__host__ __device__ inline
complex<double> ctanh(const complex<double>& z){
double x, y;
double t, beta, s, rho, denom;
uint32_t hx, ix, lx;

x = z.real();
y = z.imag();

extract_words(hx, lx, x);
ix = hx & 0x7fffffff;


if (ix >= 0x7ff00000) {
if ((ix & 0xfffff) | lx)	
return (complex<double>(x, (y == 0 ? y : x * y)));
set_high_word(x, hx - 0x40000000);	
return (complex<double>(x, copysign(0.0, isinf(y) ? y : sin(y) * cos(y))));
}


if (!isfinite(y))
return (complex<double>(y - y, y - y));


if (ix >= 0x40360000) {	
double exp_mx = exp(-fabs(x));
return (complex<double>(copysign(1.0, x),
4.0 * sin(y) * cos(y) * exp_mx * exp_mx));
}


t = tan(y);
beta = 1.0 + t * t;	
s = sinh(x);
rho = sqrt(1.0 + s * s);	
denom = 1.0 + beta * s * s;
return (complex<double>((beta * rho * s) / denom, t / denom));
}

__host__ __device__ inline
complex<double> ctan(complex<double> z){

z = ctanh(complex<double>(-z.imag(), z.real()));
return (complex<double>(z.imag(), -z.real()));
}

} 

} 


template <typename ValueType>
__host__ __device__
inline complex<ValueType> tan(const complex<ValueType>& z){
return sin(z)/cos(z);
}

template <typename ValueType>
__host__ __device__
inline complex<ValueType> tanh(const complex<ValueType>& z){
return (hydra_thrust::exp(ValueType(2)*z)-ValueType(1))/
(hydra_thrust::exp(ValueType(2)*z)+ValueType(1));
}

template <>
__host__ __device__
inline complex<double> tan(const complex<double>& z){
return detail::complex::ctan(z);
}

template <>
__host__ __device__
inline complex<double> tanh(const complex<double>& z){
return detail::complex::ctanh(z);
}

} 
