





#pragma once

#include <hydra/detail/external/hydra_thrust/complex.h>
#include <hydra/detail/external/hydra_thrust/detail/complex/math_private.h>
#include <cfloat>
#include <cmath>

namespace hydra_thrust{
namespace detail{
namespace complex{		      	

using hydra_thrust::complex;

__host__ __device__ inline
complex<float> clog_for_large_values(complex<float> z);



__host__ __device__
inline float
f(float a, float b, float hypot_a_b)
{
if (b < 0.0f)
return ((hypot_a_b - b) / 2.0f);
if (b == 0.0f)
return (a / 2.0f);
return (a * a / (hypot_a_b + b) / 2.0f);
}


__host__ __device__ 
inline void
do_hard_work(float x, float y, float *rx, int *B_is_usable, float *B,
float *sqrt_A2my2, float *new_y)
{
float R, S, A; 
float Am1, Amy; 
const float A_crossover = 10; 
const float FOUR_SQRT_MIN = 4.336808689942017736029811e-19f;; 
const float B_crossover = 0.6417f; 
R = hypotf(x, y + 1);
S = hypotf(x, y - 1);

A = (R + S) / 2;
if (A < 1)
A = 1;

if (A < A_crossover) {
if (y == 1 && x < FLT_EPSILON * FLT_EPSILON / 128) {
*rx = sqrtf(x);
} else if (x >= FLT_EPSILON * fabsf(y - 1)) {
Am1 = f(x, 1 + y, R) + f(x, 1 - y, S);
*rx = log1pf(Am1 + sqrtf(Am1 * (A + 1)));
} else if (y < 1) {
*rx = x / sqrtf((1 - y) * (1 + y));
} else {
*rx = log1pf((y - 1) + sqrtf((y - 1) * (y + 1)));
}
} else {
*rx = logf(A + sqrtf(A * A - 1));
}

*new_y = y;

if (y < FOUR_SQRT_MIN) {
*B_is_usable = 0;
*sqrt_A2my2 = A * (2 / FLT_EPSILON);
*new_y = y * (2 / FLT_EPSILON);
return;
}

*B = y / A;
*B_is_usable = 1;

if (*B > B_crossover) {
*B_is_usable = 0;
if (y == 1 && x < FLT_EPSILON / 128) {
*sqrt_A2my2 = sqrtf(x) * sqrtf((A + y) / 2);
} else if (x >= FLT_EPSILON * fabsf(y - 1)) {
Amy = f(x, y + 1, R) + f(x, y - 1, S);
*sqrt_A2my2 = sqrtf(Amy * (A + y));
} else if (y > 1) {
*sqrt_A2my2 = x * (4 / FLT_EPSILON / FLT_EPSILON) * y /
sqrtf((y + 1) * (y - 1));
*new_y = y * (4 / FLT_EPSILON / FLT_EPSILON);
} else {
*sqrt_A2my2 = sqrtf((1 - y) * (1 + y));
}
}

}

__host__ __device__ inline
complex<float>
casinhf(complex<float> z)
{
float x, y, ax, ay, rx, ry, B, sqrt_A2my2, new_y;
int B_is_usable;
complex<float> w;
const float RECIP_EPSILON = 1.0 / FLT_EPSILON;
const float m_ln2 = 6.9314718055994531e-1f; 
x = z.real();
y = z.imag();
ax = fabsf(x);
ay = fabsf(y);

if (isnan(x) || isnan(y)) {
if (isinf(x))
return (complex<float>(x, y + y));
if (isinf(y))
return (complex<float>(y, x + x));
if (y == 0)
return (complex<float>(x + x, y));
return (complex<float>(x + 0.0f + (y + 0), x + 0.0f + (y + 0)));
}

if (ax > RECIP_EPSILON || ay > RECIP_EPSILON) {
if (signbit(x) == 0)
w = clog_for_large_values(z) + m_ln2;
else
w = clog_for_large_values(-z) + m_ln2;
return (complex<float>(copysignf(w.real(), x),
copysignf(w.imag(), y)));
}

if (x == 0 && y == 0)
return (z);

raise_inexact();

const float SQRT_6_EPSILON = 8.4572793338e-4f;	
if (ax < SQRT_6_EPSILON / 4 && ay < SQRT_6_EPSILON / 4)
return (z);

do_hard_work(ax, ay, &rx, &B_is_usable, &B, &sqrt_A2my2, &new_y);
if (B_is_usable)
ry = asinf(B);
else
ry = atan2f(new_y, sqrt_A2my2);
return (complex<float>(copysignf(rx, x), copysignf(ry, y)));
}

__host__ __device__ inline
complex<float> casinf(complex<float> z)
{
complex<float> w = casinhf(complex<float>(z.imag(), z.real()));

return (complex<float>(w.imag(), w.real()));
}

__host__ __device__ inline
complex<float> cacosf(complex<float> z)
{
float x, y, ax, ay, rx, ry, B, sqrt_A2mx2, new_x;
int sx, sy;
int B_is_usable;
complex<float> w;
const float pio2_hi = 1.5707963267948966e0f; 
const volatile float pio2_lo = 6.1232339957367659e-17f;	
const float m_ln2 = 6.9314718055994531e-1f; 

x = z.real();
y = z.imag();
sx = signbit(x);
sy = signbit(y);
ax = fabsf(x);
ay = fabsf(y);

if (isnan(x) || isnan(y)) {
if (isinf(x))
return (complex<float>(y + y, -infinity<float>()));
if (isinf(y))
return (complex<float>(x + x, -y));
if (x == 0)
return (complex<float>(pio2_hi + pio2_lo, y + y));
return (complex<float>(x + 0.0f + (y + 0), x + 0.0f + (y + 0)));
}

const float RECIP_EPSILON = 1.0 / FLT_EPSILON;
if (ax > RECIP_EPSILON || ay > RECIP_EPSILON) {
w = clog_for_large_values(z);
rx = fabsf(w.imag());
ry = w.real() + m_ln2;
if (sy == 0)
ry = -ry;
return (complex<float>(rx, ry));
}

if (x == 1 && y == 0)
return (complex<float>(0, -y));

raise_inexact();

const float SQRT_6_EPSILON = 8.4572793338e-4f;	
if (ax < SQRT_6_EPSILON / 4 && ay < SQRT_6_EPSILON / 4)
return (complex<float>(pio2_hi - (x - pio2_lo), -y));

do_hard_work(ay, ax, &ry, &B_is_usable, &B, &sqrt_A2mx2, &new_x);
if (B_is_usable) {
if (sx == 0)
rx = acosf(B);
else
rx = acosf(-B);
} else {
if (sx == 0)
rx = atan2f(sqrt_A2mx2, new_x);
else
rx = atan2f(sqrt_A2mx2, -new_x);
}
if (sy == 0)
ry = -ry;
return (complex<float>(rx, ry));
}

__host__ __device__ inline
complex<float> cacoshf(complex<float> z)
{
complex<float> w;
float rx, ry;

w = cacosf(z);
rx = w.real();
ry = w.imag();

if (isnan(rx) && isnan(ry))
return (complex<float>(ry, rx));


if (isnan(rx))
return (complex<float>(fabsf(ry), rx));

if (isnan(ry))
return (complex<float>(ry, ry));
return (complex<float>(fabsf(ry), copysignf(rx, z.imag())));
}


__host__ __device__ inline
complex<float> clog_for_large_values(complex<float> z)
{
float x, y;
float ax, ay, t;
const float m_e = 2.7182818284590452e0f; 

x = z.real();
y = z.imag();
ax = fabsf(x);
ay = fabsf(y);
if (ax < ay) {
t = ax;
ax = ay;
ay = t;
}

if (ax > FLT_MAX / 2)
return (complex<float>(logf(hypotf(x / m_e, y / m_e)) + 1,
atan2f(y, x)));

const float QUARTER_SQRT_MAX = 2.3058430092136939520000000e+18f; 
const float SQRT_MIN =	1.084202172485504434007453e-19f; 
if (ax > QUARTER_SQRT_MAX || ay < SQRT_MIN)
return (complex<float>(logf(hypotf(x, y)), atan2f(y, x)));

return (complex<float>(logf(ax * ax + ay * ay) / 2, atan2f(y, x)));
}




__host__ __device__
inline float sum_squares(float x, float y)
{
const float SQRT_MIN =	1.084202172485504434007453e-19f; 

if (y < SQRT_MIN)
return (x * x);

return (x * x + y * y);
}

__host__ __device__
inline float real_part_reciprocal(float x, float y)
{
float scale;
uint32_t hx, hy;
int32_t ix, iy;

get_float_word(hx, x);
ix = hx & 0x7f800000;
get_float_word(hy, y);
iy = hy & 0x7f800000;
const int BIAS = FLT_MAX_EXP - 1;
const int CUTOFF = (FLT_MANT_DIG / 2 + 1);
if (ix - iy >= CUTOFF << 23 || isinf(x))
return (1 / x);
if (iy - ix >= CUTOFF << 23)
return (x / y / y);
if (ix <= (BIAS + FLT_MAX_EXP / 2 - CUTOFF) << 23)
return (x / (x * x + y * y));
set_float_word(scale, 0x7f800000 - ix);
x *= scale;
y *= scale;
return (x / (x * x + y * y) * scale);
}

#if __cplusplus >= 201103L || !defined _MSC_VER
__host__ __device__ inline
complex<float> catanhf(complex<float> z)
{
float x, y, ax, ay, rx, ry;
const volatile float pio2_lo = 6.1232339957367659e-17; 
const float pio2_hi = 1.5707963267948966e0;


x = z.real();
y = z.imag();
ax = fabsf(x);
ay = fabsf(y);


if (y == 0 && ax <= 1)
return (complex<float>(atanhf(x), y));

if (x == 0)
return (complex<float>(x, atanf(y)));

if (isnan(x) || isnan(y)) {
if (isinf(x))
return (complex<float>(copysignf(0, x), y + y));
if (isinf(y))
return (complex<float>(copysignf(0, x),
copysignf(pio2_hi + pio2_lo, y)));
return (complex<float>(x + 0.0f + (y + 0.0f), x + 0.0f + (y + 0.0f)));
}

const float RECIP_EPSILON = 1.0f / FLT_EPSILON;
if (ax > RECIP_EPSILON || ay > RECIP_EPSILON)
return (complex<float>(real_part_reciprocal(x, y),
copysignf(pio2_hi + pio2_lo, y)));

const float SQRT_3_EPSILON = 5.9801995673e-4; 
if (ax < SQRT_3_EPSILON / 2 && ay < SQRT_3_EPSILON / 2) {
raise_inexact();
return (z);
}

const float m_ln2 = 6.9314718056e-1f; 
if (ax == 1 && ay < FLT_EPSILON)
rx = (m_ln2 - logf(ay)) / 2;
else
rx = log1pf(4 * ax / sum_squares(ax - 1, ay)) / 4;

if (ax == 1)
ry = atan2f(2, -ay) / 2;
else if (ay < FLT_EPSILON)
ry = atan2f(2 * ay, (1 - ax) * (1 + ax)) / 2;
else
ry = atan2f(2 * ay, (1 - ax) * (1 + ax) - ay * ay) / 2;

return (complex<float>(copysignf(rx, x), copysignf(ry, y)));
}

__host__ __device__ inline
complex<float>catanf(complex<float> z){
complex<float> w = catanhf(complex<float>(z.imag(), z.real()));
return (complex<float>(w.imag(), w.real()));
}
#endif

} 

} 


template <>
__host__ __device__
inline complex<float> acos(const complex<float>& z){
return detail::complex::cacosf(z);
}

template <>
__host__ __device__
inline complex<float> asin(const complex<float>& z){
return detail::complex::casinf(z);
}

#if __cplusplus >= 201103L || !defined _MSC_VER
template <>
__host__ __device__
inline complex<float> atan(const complex<float>& z){
return detail::complex::catanf(z);
}
#endif

template <>
__host__ __device__
inline complex<float> acosh(const complex<float>& z){
return detail::complex::cacoshf(z);
}


template <>
__host__ __device__
inline complex<float> asinh(const complex<float>& z){
return detail::complex::casinhf(z);
}

#if __cplusplus >= 201103L || !defined _MSC_VER
template <>
__host__ __device__
inline complex<float> atanh(const complex<float>& z){
return detail::complex::catanhf(z);
}
#endif

} 
