





#pragma once

#include <hydra/detail/external/hydra_thrust/complex.h>
#include <hydra/detail/external/hydra_thrust/detail/complex/math_private.h>

namespace hydra_thrust{
namespace detail{
namespace complex{

__host__ __device__ inline
float frexp_expf(float x, int *expt){
const uint32_t k = 235;                 
const float kln2 =  162.88958740F;       

float exp_x;
uint32_t hx;

exp_x = expf(x - kln2);
get_float_word(hx, exp_x);
*expt = (hx >> 23) - (0x7f + 127) + k;
set_float_word(exp_x, (hx & 0x7fffff) | ((0x7f + 127) << 23));
return (exp_x);
}

__host__ __device__ inline
complex<float> 
ldexp_cexpf(complex<float> z, int expt)
{
float x, y, exp_x, scale1, scale2;
int ex_expt, half_expt;

x = z.real();
y = z.imag();
exp_x = frexp_expf(x, &ex_expt);
expt += ex_expt;

half_expt = expt / 2;
set_float_word(scale1, (0x7f + half_expt) << 23);
half_expt = expt - half_expt;
set_float_word(scale2, (0x7f + half_expt) << 23);

return (complex<float>(std::cos(y) * exp_x * scale1 * scale2,
std::sin(y) * exp_x * scale1 * scale2));
}

__host__ __device__ inline
complex<float> cexpf(const complex<float>& z){
float x, y, exp_x;
uint32_t hx, hy;

const uint32_t
exp_ovfl  = 0x42b17218,		
cexp_ovfl = 0x43400074;		

x = z.real();
y = z.imag();

get_float_word(hy, y);
hy &= 0x7fffffff;


if (hy == 0)
return (complex<float>(std::exp(x), y));
get_float_word(hx, x);

if ((hx & 0x7fffffff) == 0){
return (complex<float>(std::cos(y), std::sin(y)));
}
if (hy >= 0x7f800000) {
if ((hx & 0x7fffffff) != 0x7f800000) {

return (complex<float>(y - y, y - y));
} else if (hx & 0x80000000) {

return (complex<float>(0.0, 0.0));
} else {

return (complex<float>(x, y - y));
}
}

if (hx >= exp_ovfl && hx <= cexp_ovfl) {

return (ldexp_cexpf(z, 0));
} else {

exp_x = std::exp(x);
return (complex<float>(exp_x * std::cos(y), exp_x * std::sin(y)));
}
}

} 

} 

template <>
__host__ __device__
inline complex<float> exp(const complex<float>& z){    
return detail::complex::cexpf(z);
}    

} 
