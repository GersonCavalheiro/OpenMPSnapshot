






#pragma once

#include <hydra/detail/external/hydra_thrust/complex.h>
#include <hydra/detail/external/hydra_thrust/detail/complex/math_private.h>

namespace hydra_thrust{
namespace detail{
namespace complex{

using hydra_thrust::complex;


__host__ __device__ inline
double trim(double x){
uint32_t hi;    
get_high_word(hi, x);
insert_words(x, hi &0xfffffff8, 0);
return x;
}


__host__ __device__ inline
complex<double> clog(const complex<double>& z){

double x, y;
double ax, ay;
double x0, y0, x1, y1, x2, y2, t, hm1;
double val[12];
int i, sorted;
const double e = 2.7182818284590452354;

x = z.real();
y = z.imag();


if (x != x || y != y){
return (complex<double>(std::log(norm(z)), std::atan2(y, x)));
}

ax = std::abs(x);
ay = std::abs(y);
if (ax < ay) {
t = ax;
ax = ay;
ay = t;
}


if (ay > 5e307){
return (complex<double>(std::log(hypot(x / e, y / e)) + 1.0, std::atan2(y, x)));
}
if (ax == 1.) {
if (ay < 1e-150){
return (complex<double>((ay * 0.5) * ay, std::atan2(y, x)));
}
return (complex<double>(log1p(ay * ay) * 0.5, std::atan2(y, x)));
}


if (ax < 1e-50 || ay < 1e-50 || ax > 1e50 || ay > 1e50){
return (complex<double>(std::log(hypot(x, y)), std::atan2(y, x)));
}





if (ax >= 1.0){
return (complex<double>(log1p((ax-1)*(ax+1) + ay*ay) * 0.5, atan2(y, x)));
}

if (ax*ax + ay*ay <= 0.7){
return (complex<double>(std::log(ax*ax + ay*ay) * 0.5, std::atan2(y, x)));
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
return (complex<double>(0.5 * log1p(hm1), atan2(y, x)));
}

} 

} 

template <typename ValueType>
__host__ __device__
inline complex<ValueType> log(const complex<ValueType>& z){
return complex<ValueType>(std::log(hydra_thrust::abs(z)),hydra_thrust::arg(z));
}

template <>
__host__ __device__
inline complex<double> log(const complex<double>& z){
return detail::complex::clog(z);
}

template <typename ValueType>
__host__ __device__
inline complex<ValueType> log10(const complex<ValueType>& z){ 
return hydra_thrust::log(z)/ValueType(2.30258509299404568402);
}

} 

