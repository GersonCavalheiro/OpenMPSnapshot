






#pragma once

#include <hydra/detail/external/hydra_thrust/complex.h>
#include <hydra/detail/external/hydra_thrust/detail/complex/math_private.h>

namespace hydra_thrust{
namespace detail{
namespace complex{		      	

using hydra_thrust::complex;

__host__ __device__ inline
complex<float> ccoshf(const complex<float>& z){
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
if (iy == 0){
return (complex<float>(coshf(x), x * y));
}
if (ix < 0x41100000){	
return (complex<float>(coshf(x) * cosf(y), sinhf(x) * sinf(y)));
}

if (ix < 0x42b17218) {

h = expf(fabsf(x)) * 0.5f;
return (complex<float>(h * cosf(y), copysignf(h, x) * sinf(y)));
} else if (ix < 0x4340b1e7) {

hydra_thrust::complex<float> z_;
z_ = ldexp_cexpf(complex<float>(fabsf(x), y), -1);
return (complex<float>(z_.real(), z_.imag() * copysignf(1.0f, x)));
} else {

h = huge * x;
return (complex<float>(h * h * cosf(y), h * sinf(y)));
}
}

if (ix == 0 && iy >= 0x7f800000){
return (complex<float>(y - y, copysignf(0.0f, x * (y - y))));
}
if (iy == 0 && ix >= 0x7f800000) {
if ((hx & 0x7fffff) == 0)
return (complex<float>(x * x, copysignf(0.0f, x) * y));
return (complex<float>(x * x, copysignf(0.0f, (x + x) * y)));
}

if (ix < 0x7f800000 && iy >= 0x7f800000){
return (complex<float>(y - y, x * (y - y)));
}

if (ix >= 0x7f800000 && (hx & 0x7fffff) == 0) {
if (iy >= 0x7f800000)
return (complex<float>(x * x, x * (y - y)));
return (complex<float>((x * x) * cosf(y), x * sinf(y)));
}
return (complex<float>((x * x) * (y - y), (x + x) * (y - y)));
}

__host__ __device__ inline
complex<float> ccosf(const complex<float>& z){	
return (ccoshf(complex<float>(-z.imag(), z.real())));
}

} 

} 

template <>
__host__ __device__
inline complex<float> cos(const complex<float>& z){
return detail::complex::ccosf(z);
}

template <>
__host__ __device__
inline complex<float> cosh(const complex<float>& z){
return detail::complex::ccoshf(z);
}

} 
