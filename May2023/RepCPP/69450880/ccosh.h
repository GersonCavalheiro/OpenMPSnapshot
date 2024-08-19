





#pragma once

#include <hydra/detail/external/hydra_thrust/complex.h>
#include <hydra/detail/external/hydra_thrust/detail/complex/math_private.h>

namespace hydra_thrust{
namespace detail{
namespace complex{		      	



__host__ __device__ inline
hydra_thrust::complex<double> ccosh(const hydra_thrust::complex<double>& z){


const double huge = 8.98846567431157953864652595395e+307; 
double x, y, h;
uint32_t hx, hy, ix, iy, lx, ly;

x = z.real();
y = z.imag();

extract_words(hx, lx, x);
extract_words(hy, ly, y);

ix = 0x7fffffff & hx;
iy = 0x7fffffff & hy;


if (ix < 0x7ff00000 && iy < 0x7ff00000) {
if ((iy | ly) == 0)
return (hydra_thrust::complex<double>(::cosh(x), x * y));
if (ix < 0x40360000)	
return (hydra_thrust::complex<double>(::cosh(x) * ::cos(y), ::sinh(x) * ::sin(y)));


if (ix < 0x40862e42) {

h = ::exp(::fabs(x)) * 0.5;
return (hydra_thrust::complex<double>(h * cos(y), copysign(h, x) * sin(y)));
} else if (ix < 0x4096bbaa) {

hydra_thrust::complex<double> z_;
z_ = ldexp_cexp(hydra_thrust::complex<double>(fabs(x), y), -1);
return (hydra_thrust::complex<double>(z_.real(), z_.imag() * copysign(1.0, x)));
} else {

h = huge * x;
return (hydra_thrust::complex<double>(h * h * cos(y), h * sin(y)));
}
}


if ((ix | lx) == 0 && iy >= 0x7ff00000)
return (hydra_thrust::complex<double>(y - y, copysign(0.0, x * (y - y))));


if ((iy | ly) == 0 && ix >= 0x7ff00000) {
if (((hx & 0xfffff) | lx) == 0)
return (hydra_thrust::complex<double>(x * x, copysign(0.0, x) * y));
return (hydra_thrust::complex<double>(x * x, copysign(0.0, (x + x) * y)));
}


if (ix < 0x7ff00000 && iy >= 0x7ff00000)
return (hydra_thrust::complex<double>(y - y, x * (y - y)));


if (ix >= 0x7ff00000 && ((hx & 0xfffff) | lx) == 0) {
if (iy >= 0x7ff00000)
return (hydra_thrust::complex<double>(x * x, x * (y - y)));
return (hydra_thrust::complex<double>((x * x) * cos(y), x * sin(y)));
}


return (hydra_thrust::complex<double>((x * x) * (y - y), (x + x) * (y - y)));
}


__host__ __device__ inline
hydra_thrust::complex<double> ccos(const hydra_thrust::complex<double>& z){	

return (ccosh(hydra_thrust::complex<double>(-z.imag(), z.real())));
}

} 

} 

template <typename ValueType>
__host__ __device__
inline complex<ValueType> cos(const complex<ValueType>& z){
const ValueType re = z.real();
const ValueType im = z.imag();
return complex<ValueType>(std::cos(re) * std::cosh(im), 
-std::sin(re) * std::sinh(im));
}

template <typename ValueType>
__host__ __device__
inline complex<ValueType> cosh(const complex<ValueType>& z){
const ValueType re = z.real();
const ValueType im = z.imag();
return complex<ValueType>(std::cosh(re) * std::cos(im), 
std::sinh(re) * std::sin(im));
}

template <>
__host__ __device__
inline hydra_thrust::complex<double> cos(const hydra_thrust::complex<double>& z){
return detail::complex::ccos(z);
}

template <>
__host__ __device__
inline hydra_thrust::complex<double> cosh(const hydra_thrust::complex<double>& z){
return detail::complex::ccosh(z);
}

} 
