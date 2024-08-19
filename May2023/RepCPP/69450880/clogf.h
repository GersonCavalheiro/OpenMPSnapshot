





#pragma once

#include <hydra/detail/external/hydra_thrust/complex.h>
#include <hydra/detail/external/hydra_thrust/detail/complex/math_private.h>

namespace hydra_thrust{
namespace detail{
namespace complex{

using hydra_thrust::complex;


__host__ __device__ inline
float trim(float x){
uint32_t hx;
get_float_word(hx, x);
hx &= 0xffff0000;
float ret;
set_float_word(ret,hx);
return ret;
}


__host__ __device__ inline
complex<float> clogf(const complex<float>& z){

float x, y;
float ax, ay;
float x0, y0, x1, y1, x2, y2, t, hm1;
float val[12];
int i, sorted;	
const float e = 2.7182818284590452354f;

x = z.real();
y = z.imag();


if (x != x || y != y){
return (complex<float>(std::log(norm(z)), std::atan2(y, x)));
}

ax = std::abs(x);
ay = std::abs(y);
if (ax < ay) {
t = ax;
ax = ay;
ay = t;
}


if (ay > 1e34f){ 
return (complex<float>(std::log(hypotf(x / e, y / e)) + 1.0f, std::atan2(y, x)));
}
if (ax == 1.f) {
if (ay < 1e-19f){
return (complex<float>((ay * 0.5f) * ay, std::atan2(y, x)));
}
return (complex<float>(log1pf(ay * ay) * 0.5f, std::atan2(y, x)));
}


if (ax < 1e-6f || ay < 1e-6f || ax > 1e6f || ay > 1e6f){
return (complex<float>(std::log(hypotf(x, y)), std::atan2(y, x)));
}





if (ax >= 1.0f){
return (complex<float>(log1pf((ax-1.f)*(ax+1.f) + ay*ay) * 0.5f, atan2(y, x)));
}

if (ax*ax + ay*ay <= 0.7f){
return (complex<float>(std::log(ax*ax + ay*ay) * 0.5f, std::atan2(y, x)));
}




x0 = trim(ax);
ax = ax-x0;
x1 = trim(ax);
x2 = ax-x1;
y0 = trim(ay);
ay = ay-y0;
y1 = trim(ay);
y2 = ay-y1;

val[0] = x0*x0;
val[1] = y0*y0;
val[2] = 2*x0*x1;
val[3] = 2*y0*y1;
val[4] = x1*x1;
val[5] = y1*y1;
val[6] = 2*x0*x2;
val[7] = 2*y0*y2;
val[8] = 2*x1*x2;
val[9] = 2*y1*y2;
val[10] = x2*x2;
val[11] = y2*y2;



do {
sorted = 1;
for (i=0;i<11;i++) {
if (val[i] < val[i+1]) {
sorted = 0;
t = val[i];
val[i] = val[i+1];
val[i+1] = t;
}
}
} while (!sorted);

hm1 = -1;
for (i=0;i<12;i++){
hm1 += val[i];
}
return (complex<float>(0.5f * log1pf(hm1), atan2(y, x)));
}

} 

} 

template <>
__host__ __device__
inline complex<float> log(const complex<float>& z){
return detail::complex::clogf(z);
}

} 

