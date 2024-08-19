





#pragma once

#include <hydra/detail/external/hydra_thrust/complex.h>
#include <hydra/detail/external/hydra_thrust/detail/complex/math_private.h>

namespace hydra_thrust{
namespace detail{
namespace complex{		      	

__host__ __device__ inline
double frexp_exp(double x, int *expt){
const uint32_t k = 1799;		
const double kln2 =  1246.97177782734161156;	

double exp_x;
uint32_t hx;


exp_x = exp(x - kln2);
get_high_word(hx, exp_x);
*expt = (hx >> 20) - (0x3ff + 1023) + k;
set_high_word(exp_x, (hx & 0xfffff) | ((0x3ff + 1023) << 20));
return (exp_x);
}


__host__ __device__ inline
complex<double>	ldexp_cexp(complex<double> z, int expt){
double x, y, exp_x, scale1, scale2;
int ex_expt, half_expt;

x = z.real();
y = z.imag();
exp_x = frexp_exp(x, &ex_expt);
expt += ex_expt;


half_expt = expt / 2;
insert_words(scale1, (0x3ff + half_expt) << 20, 0);
half_expt = expt - half_expt;
insert_words(scale2, (0x3ff + half_expt) << 20, 0);

return (complex<double>(cos(y) * exp_x * scale1 * scale2,
sin(y) * exp_x * scale1 * scale2));
}


__host__ __device__ inline
complex<double> cexp(const complex<double>& z){
double x, y, exp_x;
uint32_t hx, hy, lx, ly;

const uint32_t
exp_ovfl  = 0x40862e42,			
cexp_ovfl = 0x4096b8e4;			


x = z.real();
y = z.imag();

extract_words(hy, ly, y);
hy &= 0x7fffffff;


if ((hy | ly) == 0)
return (complex<double>(exp(x), y));
extract_words(hx, lx, x);

if (((hx & 0x7fffffff) | lx) == 0)
return (complex<double>(cos(y), sin(y)));

if (hy >= 0x7ff00000) {
if (lx != 0 || (hx & 0x7fffffff) != 0x7ff00000) {

return (complex<double>(y - y, y - y));
} else if (hx & 0x80000000) {

return (complex<double>(0.0, 0.0));
} else {

return (complex<double>(x, y - y));
}
}

if (hx >= exp_ovfl && hx <= cexp_ovfl) {

return (ldexp_cexp(z, 0));
} else {

exp_x = std::exp(x);
return (complex<double>(exp_x * cos(y), exp_x * sin(y)));
}
}

} 

} 

template <typename ValueType>
__host__ __device__
inline complex<ValueType> exp(const complex<ValueType>& z){    
return polar(std::exp(z.real()),z.imag());
}

template <>
__host__ __device__
inline complex<double> exp(const complex<double>& z){    
return detail::complex::cexp(z);
}

} 
