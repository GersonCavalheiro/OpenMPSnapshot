






#pragma once

#include <hydra/detail/external/hydra_thrust/complex.h>
#include <hydra/detail/external/hydra_thrust/detail/complex/math_private.h>
#include <cmath>

namespace hydra_thrust{
namespace detail{
namespace complex{		      	

using hydra_thrust::complex;

__host__ __device__ inline
complex<double> csqrt(const complex<double>& z){
complex<double> result;
double a, b;
double t;
int scale;


const double THRESH = 7.446288774449766337959726e+307;

a = z.real();
b = z.imag();


if (z == 0.0)
return (complex<double>(0.0, b));
if (isinf(b))
return (complex<double>(infinity<double>(), b));
if (isnan(a)) {
t = (b - b) / (b - b);	
return (complex<double>(a, t));	
}
if (isinf(a)) {

if (signbit(a))
return (complex<double>(fabs(b - b), copysign(a, b)));
else
return (complex<double>(a, copysign(b - b, b)));
}


const double low_thresh = 4.450147717014402766180465e-308;
scale = 0;

if (fabs(a) >= THRESH || fabs(b) >= THRESH) {

a *= 0.25;
b *= 0.25;
scale = 1;
}else if (fabs(a) <= low_thresh && fabs(b) <= low_thresh) {

a *= 4.0;
b *= 4.0;
scale = 2;
}



if (a >= 0.0) {
t = sqrt((a + hypot(a, b)) * 0.5);
result = complex<double>(t, b / (2 * t));
} else {
t = sqrt((-a + hypot(a, b)) * 0.5);
result = complex<double>(fabs(b) / (2 * t), copysign(t, b));
}


if (scale == 1)
return (result * 2.0);
else if (scale == 2)
return (result * 0.5);
else
return (result);
}

} 

} 

template <typename ValueType>
__host__ __device__
inline complex<ValueType> sqrt(const complex<ValueType>& z){
return hydra_thrust::polar(std::sqrt(hydra_thrust::abs(z)),hydra_thrust::arg(z)/ValueType(2));
}

template <>
__host__ __device__
inline complex<double> sqrt(const complex<double>& z){
return detail::complex::csqrt(z);
}

} 
