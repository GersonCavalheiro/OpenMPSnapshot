






#pragma once

#include <hydra/detail/external/hydra_thrust/complex.h>
#include <hydra/detail/external/hydra_thrust/detail/complex/math_private.h>
#include <cmath>

namespace hydra_thrust{
namespace detail{
namespace complex{		      	

using hydra_thrust::complex;

__host__ __device__ inline
complex<float> csqrtf(const complex<float>& z){
float a = z.real(), b = z.imag();
float t;
int scale;
complex<float> result;


const float THRESH = 1.40949553037932e+38f;


if (z == 0.0f)
return (complex<float>(0, b));
if (isinf(b))
return (complex<float>(infinity<float>(), b));
if (isnan(a)) {
t = (b - b) / (b - b);	
return (complex<float>(a, t));	
}
if (isinf(a)) {

if (signbit(a))
return (complex<float>(fabsf(b - b), copysignf(a, b)));
else
return (complex<float>(a, copysignf(b - b, b)));
}




const float low_thresh = 2.35098870164458e-38f;
scale = 0;

if (fabsf(a) >= THRESH || fabsf(b) >= THRESH) {

a *= 0.25f;
b *= 0.25f;
scale = 1;
}else if (fabsf(a) <= low_thresh && fabsf(b) <= low_thresh) {

a *= 4.f;
b *= 4.f;
scale = 2;
}


if (a >= 0.0f) {
t = sqrtf((a + hypotf(a, b)) * 0.5f);
result = complex<float>(t, b / (2.0f * t));
} else {
t = sqrtf((-a + hypotf(a, b)) * 0.5f);
result = complex<float>(fabsf(b) / (2.0f * t), copysignf(t, b));
}


if (scale == 1)
return (result * 2.0f);
else if (scale == 2)
return (result * 0.5f);
else
return (result);
}      

} 

} 

template <>
__host__ __device__
inline complex<float> sqrt(const complex<float>& z){
return detail::complex::csqrtf(z);
}

} 
