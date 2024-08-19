

#pragma once

#include <hydra/detail/external/hydra_thrust/complex.h>
#include <hydra/detail/external/hydra_thrust/detail/complex/math_private.h>
#include <cmath>

namespace hydra_thrust{
namespace detail{
namespace complex{	 
__host__ __device__
inline complex<float> cprojf(const complex<float>& z){
if(!isinf(z.real()) && !isinf(z.imag())){
return z;
}else{
return complex<float>(infinity<float>(), copysignf(0.0, z.imag()));
}
}

__host__ __device__
inline complex<double> cproj(const complex<double>& z){
if(!isinf(z.real()) && !isinf(z.imag())){
return z;
}else{
return complex<double>(infinity<double>(), copysign(0.0, z.imag()));
}
}

}

}

template <typename T>
__host__ __device__
inline hydra_thrust::complex<T> proj(const hydra_thrust::complex<T>& z){
return detail::complex::cproj(z);
}


template <>
__host__ __device__
inline hydra_thrust::complex<double> proj(const hydra_thrust::complex<double>& z){
return detail::complex::cproj(z);
}

template <>
__host__ __device__
inline hydra_thrust::complex<float> proj(const hydra_thrust::complex<float>& z){
return detail::complex::cprojf(z);
}

}

