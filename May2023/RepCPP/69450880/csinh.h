






#pragma once

#include <hydra/detail/external/hydra_thrust/complex.h>
#include <hydra/detail/external/hydra_thrust/detail/complex/math_private.h>

namespace hydra_thrust{
namespace detail{
namespace complex{		      	

using hydra_thrust::complex;

__host__ __device__ inline
complex<double> csinh(const complex<double>& z){
double x, y, h;
uint32_t hx, hy, ix, iy, lx, ly;
const double huge = 8.98846567431157953864652595395e+307; 

x = z.real();
y = z.imag();

extract_words(hx, lx, x);
extract_words(hy, ly, y);

ix = 0x7fffffff & hx;
iy = 0x7fffffff & hy;


if (ix < 0x7ff00000 && iy < 0x7ff00000) {
if ((iy | ly) == 0)
return (complex<double>(sinh(x), y));
if (ix < 0x40360000)	
return (complex<double>(sinh(x) * cos(y), cosh(x) * sin(y)));


if (ix < 0x40862e42) {

h = exp(fabs(x)) * 0.5;
return (complex<double>(copysign(h, x) * cos(y), h * sin(y)));
} else if (ix < 0x4096bbaa) {

complex<double> z_ = ldexp_cexp(complex<double>(fabs(x), y), -1);
return (complex<double>(z_.real() * copysign(1.0, x), z_.imag()));
} else {

h = huge * x;
return (complex<double>(h * cos(y), h * h * sin(y)));
}
}


if ((ix | lx) == 0 && iy >= 0x7ff00000)
return (complex<double>(copysign(0.0, x * (y - y)), y - y));


if ((iy | ly) == 0 && ix >= 0x7ff00000) {
if (((hx & 0xfffff) | lx) == 0)
return (complex<double>(x, y));
return (complex<double>(x, copysign(0.0, y)));
}


if (ix < 0x7ff00000 && iy >= 0x7ff00000)
return (complex<double>(y - y, x * (y - y)));


if (ix >= 0x7ff00000 && ((hx & 0xfffff) | lx) == 0) {
if (iy >= 0x7ff00000)
return (complex<double>(x * x, x * (y - y)));
return (complex<double>(x * cos(y), infinity<double>() * sin(y)));
}


return (complex<double>((x * x) * (y - y), (x + x) * (y - y)));
}

__host__ __device__ inline
complex<double> csin(complex<double> z){

z = csinh(complex<double>(-z.imag(), z.real()));
return (complex<double>(z.imag(), -z.real()));
}

} 

} 

template <typename ValueType>
__host__ __device__
inline complex<ValueType> sin(const complex<ValueType>& z){
const ValueType re = z.real();
const ValueType im = z.imag();
return complex<ValueType>(std::sin(re) * std::cosh(im), 
std::cos(re) * std::sinh(im));
}


template <typename ValueType>
__host__ __device__
inline complex<ValueType> sinh(const complex<ValueType>& z){
const ValueType re = z.real();
const ValueType im = z.imag();
return complex<ValueType>(std::sinh(re) * std::cos(im), 
std::cosh(re) * std::sin(im));
}

template <>
__host__ __device__
inline complex<double> sin(const complex<double>& z){
return detail::complex::csin(z);
}

template <>
__host__ __device__
inline complex<double> sinh(const complex<double>& z){
return detail::complex::csinh(z);
}

} 
