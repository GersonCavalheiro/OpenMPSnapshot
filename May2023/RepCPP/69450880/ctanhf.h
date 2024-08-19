







#pragma once

#include <hydra/detail/external/hydra_thrust/complex.h>
#include <hydra/detail/external/hydra_thrust/detail/complex/math_private.h>
#include <cmath>

namespace hydra_thrust{
namespace detail{
namespace complex{		      	

using hydra_thrust::complex;

__host__ __device__ inline
complex<float> ctanhf(const complex<float>& z){
float x, y;
float t, beta, s, rho, denom;
uint32_t hx, ix;

x = z.real();
y = z.imag();

get_float_word(hx, x);
ix = hx & 0x7fffffff;

if (ix >= 0x7f800000) {
if (ix & 0x7fffff)
return (complex<float>(x, (y == 0.0f ? y : x * y)));
set_float_word(x, hx - 0x40000000);
return (complex<float>(x,
copysignf(0, isinf(y) ? y : sinf(y) * cosf(y))));
}

if (!isfinite(y))
return (complex<float>(y - y, y - y));

if (ix >= 0x41300000) {	
float exp_mx = expf(-fabsf(x));
return (complex<float>(copysignf(1.0f, x),
4.0f * sinf(y) * cosf(y) * exp_mx * exp_mx));
}

t = tanf(y);
beta = 1.0f + t * t;
s = sinhf(x);
rho = sqrtf(1.0f + s * s);
denom = 1.0f + beta * s * s;
return (complex<float>((beta * rho * s) / denom, t / denom));
}

__host__ __device__ inline
complex<float> ctanf(complex<float> z){
z = ctanhf(complex<float>(-z.imag(), z.real()));
return (complex<float>(z.imag(), -z.real()));
}

} 

} 

template <>
__host__ __device__
inline complex<float> tan(const complex<float>& z){
return detail::complex::ctanf(z);
}

template <>
__host__ __device__
inline complex<float> tanh(const complex<float>& z){
return detail::complex::ctanhf(z);
}

} 
