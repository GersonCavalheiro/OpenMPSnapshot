






#pragma once

#include <hydra/detail/external/hydra_thrust/complex.h>
#include <hydra/detail/external/hydra_thrust/detail/complex/math_private.h>

namespace hydra_thrust{
namespace detail{
namespace complex{		      	

using hydra_thrust::complex;

__host__ __device__ inline
complex<float> csinhf(const complex<float>& z){

float x, y, h;
uint32_t hx, hy, ix, iy;

const float huge = 1.70141183460469231731687303716e+38; 

x = z.real();
y = z.imag();

get_float_word(hx, x);
get_float_word(hy, y);

ix = 0x7fffffff & hx;
iy = 0x7fffffff & hy;

if (ix < 0x7f800000 && iy < 0x7f800000) {
if (iy == 0)
return (complex<float>(sinhf(x), y));
if (ix < 0x41100000)	
return (complex<float>(sinhf(x) * cosf(y), coshf(x) * sinf(y)));


if (ix < 0x42b17218) {

h = expf(fabsf(x)) * 0.5f;
return (complex<float>(copysignf(h, x) * cosf(y), h * sinf(y)));
} else if (ix < 0x4340b1e7) {

complex<float> z_ = ldexp_cexpf(complex<float>(fabsf(x), y), -1);
return (complex<float>(z_.real() * copysignf(1.0f, x), z_.imag()));
} else {

h = huge * x;
return (complex<float>(h * cosf(y), h * h * sinf(y)));
}
}

if (ix == 0 && iy >= 0x7f800000)
return (complex<float>(copysignf(0, x * (y - y)), y - y));

if (iy == 0 && ix >= 0x7f800000) {
if ((hx & 0x7fffff) == 0)
return (complex<float>(x, y));
return (complex<float>(x, copysignf(0.0f, y)));
}

if (ix < 0x7f800000 && iy >= 0x7f800000)
return (complex<float>(y - y, x * (y - y)));

if (ix >= 0x7f800000 && (hx & 0x7fffff) == 0) {
if (iy >= 0x7f800000)
return (complex<float>(x * x, x * (y - y)));
return (complex<float>(x * cosf(y), infinity<float>() * sinf(y)));
}

return (complex<float>((x * x) * (y - y), (x + x) * (y - y)));
}

__host__ __device__ inline
complex<float> csinf(complex<float> z){
z = csinhf(complex<float>(-z.imag(), z.real()));
return (complex<float>(z.imag(), -z.real()));
}

} 

} 

template <>
__host__ __device__
inline complex<float> sin(const complex<float>& z){
return detail::complex::csinf(z);
}

template <>
__host__ __device__
inline complex<float> sinh(const complex<float>& z){
return detail::complex::csinhf(z);
}

} 
