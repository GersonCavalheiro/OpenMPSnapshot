





#pragma once

#include <hydra/detail/external/hydra_thrust/complex.h>
#include <hydra/detail/external/hydra_thrust/detail/complex/math_private.h>
#include <cfloat>
#include <cmath>

namespace hydra_thrust{
namespace detail{
namespace complex{		      	

using hydra_thrust::complex;

__host__ __device__
inline void raise_inexact(){
const volatile float tiny = 7.888609052210118054117286e-31;  
volatile float junk = 1 + tiny;
(void)junk;
}

__host__ __device__ inline complex<double> clog_for_large_values(complex<double> z);








__host__ __device__
inline double
f(double a, double b, double hypot_a_b)
{
if (b < 0)
return ((hypot_a_b - b) / 2);
if (b == 0)
return (a / 2);
return (a * a / (hypot_a_b + b) / 2);
}


__host__ __device__
inline void
do_hard_work(double x, double y, double *rx, int *B_is_usable, double *B,
double *sqrt_A2my2, double *new_y)
{
double R, S, A; 
double Am1, Amy; 
const double A_crossover = 10; 
const double FOUR_SQRT_MIN = 5.966672584960165394632772e-154; 
const double B_crossover = 0.6417; 

R = hypot(x, y + 1);		
S = hypot(x, y - 1);		


A = (R + S) / 2;

if (A < 1)
A = 1;

if (A < A_crossover) {

if (y == 1 && x < DBL_EPSILON * DBL_EPSILON / 128) {

*rx = sqrt(x);
} else if (x >= DBL_EPSILON * fabs(y - 1)) {

Am1 = f(x, 1 + y, R) + f(x, 1 - y, S);
*rx = log1p(Am1 + sqrt(Am1 * (A + 1)));
} else if (y < 1) {

*rx = x / sqrt((1 - y) * (1 + y));
} else {		

*rx = log1p((y - 1) + sqrt((y - 1) * (y + 1)));
}
} else {
*rx = log(A + sqrt(A * A - 1));
}

*new_y = y;

if (y < FOUR_SQRT_MIN) {

*B_is_usable = 0;
*sqrt_A2my2 = A * (2 / DBL_EPSILON);
*new_y = y * (2 / DBL_EPSILON);
return;
}


*B = y / A;
*B_is_usable = 1;

if (*B > B_crossover) {
*B_is_usable = 0;

if (y == 1 && x < DBL_EPSILON / 128) {

*sqrt_A2my2 = sqrt(x) * sqrt((A + y) / 2);
} else if (x >= DBL_EPSILON * fabs(y - 1)) {

Amy = f(x, y + 1, R) + f(x, y - 1, S);
*sqrt_A2my2 = sqrt(Amy * (A + y));
} else if (y > 1) {

*sqrt_A2my2 = x * (4 / DBL_EPSILON / DBL_EPSILON) * y /
sqrt((y + 1) * (y - 1));
*new_y = y * (4 / DBL_EPSILON / DBL_EPSILON);
} else {		

*sqrt_A2my2 = sqrt((1 - y) * (1 + y));
}
}
}


__host__ __device__ inline
complex<double> casinh(complex<double> z)
{
double x, y, ax, ay, rx, ry, B, sqrt_A2my2, new_y;
int B_is_usable;
complex<double> w;
const double RECIP_EPSILON = 1.0 / DBL_EPSILON;
const double m_ln2 = 6.9314718055994531e-1; 
x = z.real();
y = z.imag();
ax = fabs(x);
ay = fabs(y);

if (isnan(x) || isnan(y)) {

if (isinf(x))
return (complex<double>(x, y + y));

if (isinf(y))
return (complex<double>(y, x + x));

if (y == 0)
return (complex<double>(x + x, y));

return (complex<double>(x + 0.0 + (y + 0.0), x + 0.0 + (y + 0.0)));
}

if (ax > RECIP_EPSILON || ay > RECIP_EPSILON) {

if (signbit(x) == 0)
w = clog_for_large_values(z) + m_ln2;
else
w = clog_for_large_values(-z) + m_ln2;
return (complex<double>(copysign(w.real(), x), copysign(w.imag(), y)));
}


if (x == 0 && y == 0)
return (z);


raise_inexact();

const double SQRT_6_EPSILON = 3.6500241499888571e-8; 
if (ax < SQRT_6_EPSILON / 4 && ay < SQRT_6_EPSILON / 4)
return (z);

do_hard_work(ax, ay, &rx, &B_is_usable, &B, &sqrt_A2my2, &new_y);
if (B_is_usable)
ry = asin(B);
else
ry = atan2(new_y, sqrt_A2my2);
return (complex<double>(copysign(rx, x), copysign(ry, y)));
}


__host__ __device__ inline
complex<double> casin(complex<double> z)
{
complex<double> w = casinh(complex<double>(z.imag(), z.real()));

return (complex<double>(w.imag(), w.real()));
}


__host__ __device__ inline
complex<double> cacos(complex<double> z)
{
double x, y, ax, ay, rx, ry, B, sqrt_A2mx2, new_x;
int sx, sy;
int B_is_usable;
complex<double> w;
const double pio2_hi = 1.5707963267948966e0; 
const volatile double pio2_lo = 6.1232339957367659e-17;	
const double m_ln2 = 6.9314718055994531e-1; 

x = z.real();
y = z.imag();
sx = signbit(x);
sy = signbit(y);
ax = fabs(x);
ay = fabs(y);

if (isnan(x) || isnan(y)) {

if (isinf(x))
return (complex<double>(y + y, -infinity<double>()));

if (isinf(y))
return (complex<double>(x + x, -y));

if (x == 0)
return (complex<double>(pio2_hi + pio2_lo, y + y));

return (complex<double>(x + 0.0 + (y + 0), x + 0.0 + (y + 0)));
}

const double RECIP_EPSILON = 1.0 / DBL_EPSILON;
if (ax > RECIP_EPSILON || ay > RECIP_EPSILON) {

w = clog_for_large_values(z);
rx = fabs(w.imag());
ry = w.real() + m_ln2;
if (sy == 0)
ry = -ry;
return (complex<double>(rx, ry));
}


if (x == 1.0 && y == 0.0)
return (complex<double>(0, -y));


raise_inexact();

const double SQRT_6_EPSILON = 3.6500241499888571e-8; 
if (ax < SQRT_6_EPSILON / 4 && ay < SQRT_6_EPSILON / 4)
return (complex<double>(pio2_hi - (x - pio2_lo), -y));

do_hard_work(ay, ax, &ry, &B_is_usable, &B, &sqrt_A2mx2, &new_x);
if (B_is_usable) {
if (sx == 0)
rx = acos(B);
else
rx = acos(-B);
} else {
if (sx == 0)
rx = atan2(sqrt_A2mx2, new_x);
else
rx = atan2(sqrt_A2mx2, -new_x);
}
if (sy == 0)
ry = -ry;
return (complex<double>(rx, ry));
}


__host__ __device__ inline
complex<double> cacosh(complex<double> z)
{
complex<double> w;
double rx, ry;

w = cacos(z);
rx = w.real();
ry = w.imag();

if (isnan(rx) && isnan(ry))
return (complex<double>(ry, rx));


if (isnan(rx))
return (complex<double>(fabs(ry), rx));

if (isnan(ry))
return (complex<double>(ry, ry));
return (complex<double>(fabs(ry), copysign(rx, z.imag())));
}


__host__ __device__ inline
complex<double> clog_for_large_values(complex<double> z)
{
double x, y;
double ax, ay, t;
const double m_e = 2.7182818284590452e0; 

x = z.real();
y = z.imag();
ax = fabs(x);
ay = fabs(y);
if (ax < ay) {
t = ax;
ax = ay;
ay = t;
}


if (ax > DBL_MAX / 2)
return (complex<double>(log(hypot(x / m_e, y / m_e)) + 1, atan2(y, x)));


const double QUARTER_SQRT_MAX = 5.966672584960165394632772e-154; 
const double SQRT_MIN =	1.491668146240041348658193e-154; 
if (ax > QUARTER_SQRT_MAX || ay < SQRT_MIN)
return (complex<double>(log(hypot(x, y)), atan2(y, x)));

return (complex<double>(log(ax * ax + ay * ay) / 2, atan2(y, x)));
}




__host__ __device__
inline double sum_squares(double x, double y)
{
const double SQRT_MIN =	1.491668146240041348658193e-154; 

if (y < SQRT_MIN)
return (x * x);

return (x * x + y * y);
}


__host__ __device__
inline double real_part_reciprocal(double x, double y)
{
double scale;
uint32_t hx, hy;
int32_t ix, iy;


get_high_word(hx, x);
ix = hx & 0x7ff00000;
get_high_word(hy, y);
iy = hy & 0x7ff00000;
const int BIAS = DBL_MAX_EXP - 1;

const int CUTOFF = (DBL_MANT_DIG / 2 + 1);
if (ix - iy >= CUTOFF << 20 || isinf(x))
return (1 / x);		
if (iy - ix >= CUTOFF << 20)
return (x / y / y);	
if (ix <= (BIAS + DBL_MAX_EXP / 2 - CUTOFF) << 20)
return (x / (x * x + y * y));
scale = 1;
set_high_word(scale, 0x7ff00000 - ix);	
x *= scale;
y *= scale;
return (x / (x * x + y * y) * scale);
}



#if __cplusplus >= 201103L || !defined _MSC_VER
__host__ __device__ inline
complex<double> catanh(complex<double> z)
{
double x, y, ax, ay, rx, ry;
const volatile double pio2_lo = 6.1232339957367659e-17; 
const double pio2_hi = 1.5707963267948966e0;


x = z.real();
y = z.imag();
ax = fabs(x);
ay = fabs(y);


if (y == 0 && ax <= 1)
return (complex<double>(atanh(x), y));


if (x == 0)
return (complex<double>(x, atan(y)));

if (isnan(x) || isnan(y)) {

if (isinf(x))
return (complex<double>(copysign(0.0, x), y + y));

if (isinf(y))
return (complex<double>(copysign(0.0, x),
copysign(pio2_hi + pio2_lo, y)));

return (complex<double>(x + 0.0 + (y + 0), x + 0.0 + (y + 0)));
}

const double RECIP_EPSILON = 1.0 / DBL_EPSILON;
if (ax > RECIP_EPSILON || ay > RECIP_EPSILON)
return (complex<double>(real_part_reciprocal(x, y),
copysign(pio2_hi + pio2_lo, y)));

const double SQRT_3_EPSILON = 2.5809568279517849e-8; 
if (ax < SQRT_3_EPSILON / 2 && ay < SQRT_3_EPSILON / 2) {

raise_inexact();
return (z);
}

const double m_ln2 = 6.9314718055994531e-1; 
if (ax == 1 && ay < DBL_EPSILON)
rx = (m_ln2 - log(ay)) / 2;
else
rx = log1p(4 * ax / sum_squares(ax - 1, ay)) / 4;

if (ax == 1)
ry = atan2(2.0, -ay) / 2;
else if (ay < DBL_EPSILON)
ry = atan2(2 * ay, (1 - ax) * (1 + ax)) / 2;
else
ry = atan2(2 * ay, (1 - ax) * (1 + ax) - ay * ay) / 2;

return (complex<double>(copysign(rx, x), copysign(ry, y)));
}


__host__ __device__ inline
complex<double>catan(complex<double> z)
{
complex<double> w = catanh(complex<double>(z.imag(), z.real()));
return (complex<double>(w.imag(), w.real()));
}

#endif

} 

} 


template <typename ValueType>
__host__ __device__
inline complex<ValueType> acos(const complex<ValueType>& z){
const complex<ValueType> ret = hydra_thrust::asin(z);
const ValueType pi = ValueType(3.14159265358979323846);
return complex<ValueType>(pi/2 - ret.real(),-ret.imag());
}


template <typename ValueType>
__host__ __device__
inline complex<ValueType> asin(const complex<ValueType>& z){
const complex<ValueType> i(0,1);
return -i*asinh(i*z);
}

template <typename ValueType>
__host__ __device__
inline complex<ValueType> atan(const complex<ValueType>& z){
const complex<ValueType> i(0,1);
return -i*hydra_thrust::atanh(i*z);
}


template <typename ValueType>
__host__ __device__
inline complex<ValueType> acosh(const complex<ValueType>& z){
hydra_thrust::complex<ValueType> ret((z.real() - z.imag()) * (z.real() + z.imag()) - ValueType(1.0),
ValueType(2.0) * z.real() * z.imag());    
ret = hydra_thrust::sqrt(ret);
if (z.real() < ValueType(0.0)){
ret = -ret;
}
ret += z;
ret = hydra_thrust::log(ret);
if (ret.real() < ValueType(0.0)){
ret = -ret;
}
return ret;
}

template <typename ValueType>
__host__ __device__
inline complex<ValueType> asinh(const complex<ValueType>& z){
return hydra_thrust::log(hydra_thrust::sqrt(z*z+ValueType(1))+z);
}

template <typename ValueType>
__host__ __device__
inline complex<ValueType> atanh(const complex<ValueType>& z){
ValueType imag2 = z.imag() *  z.imag();   
ValueType n = ValueType(1.0) + z.real();
n = imag2 + n * n;

ValueType d = ValueType(1.0) - z.real();
d = imag2 + d * d;
complex<ValueType> ret(ValueType(0.25) * (std::log(n) - std::log(d)),0);

d = ValueType(1.0) -  z.real() * z.real() - imag2;

ret.imag(ValueType(0.5) * std::atan2(ValueType(2.0) * z.imag(), d));
return ret;
}

template <>
__host__ __device__
inline complex<double> acos(const complex<double>& z){
return detail::complex::cacos(z);
}

template <>
__host__ __device__
inline complex<double> asin(const complex<double>& z){
return detail::complex::casin(z);
}

#if __cplusplus >= 201103L || !defined _MSC_VER
template <>
__host__ __device__
inline complex<double> atan(const complex<double>& z){
return detail::complex::catan(z);
}
#endif

template <>
__host__ __device__
inline complex<double> acosh(const complex<double>& z){
return detail::complex::cacosh(z);
}


template <>
__host__ __device__
inline complex<double> asinh(const complex<double>& z){
return detail::complex::casinh(z);
}

#if __cplusplus >= 201103L || !defined _MSC_VER
template <>
__host__ __device__
inline complex<double> atanh(const complex<double>& z){
return detail::complex::catanh(z);
}
#endif

} 
