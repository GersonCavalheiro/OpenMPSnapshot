




#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/complex.h>
#include <hydra/detail/external/hydra_thrust/detail/cstdint.h>

namespace hydra_thrust{
namespace detail{
namespace complex{

using hydra_thrust::complex;

typedef union
{
float value;
uint32_t word;
} ieee_float_shape_type;

__host__ __device__
inline void get_float_word(uint32_t & i, float d){
ieee_float_shape_type gf_u;
gf_u.value = (d);
(i) = gf_u.word;
}

__host__ __device__
inline void get_float_word(int32_t & i, float d){
ieee_float_shape_type gf_u;
gf_u.value = (d);
(i) = gf_u.word;
}

__host__ __device__
inline void set_float_word(float & d, uint32_t i){
ieee_float_shape_type sf_u;
sf_u.word = (i);
(d) = sf_u.value;
}

typedef union
{
double value;
struct
{
uint32_t lsw;
uint32_t msw;
} parts;
struct
{
uint64_t w;
} xparts;
} ieee_double_shape_type;

__host__ __device__ inline
void get_high_word(uint32_t & i,double d){
ieee_double_shape_type gh_u;
gh_u.value = (d);
(i) = gh_u.parts.msw;                                   
}


__host__ __device__ inline
void set_high_word(double & d, uint32_t v){
ieee_double_shape_type sh_u;
sh_u.value = (d);
sh_u.parts.msw = (v);
(d) = sh_u.value;
}


__host__ __device__ inline 
void  insert_words(double & d, uint32_t ix0, uint32_t ix1){
ieee_double_shape_type iw_u;
iw_u.parts.msw = (ix0);
iw_u.parts.lsw = (ix1);
(d) = iw_u.value;
}


__host__ __device__ inline
void  extract_words(uint32_t & ix0,uint32_t & ix1, double d){
ieee_double_shape_type ew_u;
ew_u.value = (d);
(ix0) = ew_u.parts.msw;
(ix1) = ew_u.parts.lsw;
}


__host__ __device__ inline
void  extract_words(int32_t & ix0,int32_t & ix1, double d){
ieee_double_shape_type ew_u;
ew_u.value = (d);
(ix0) = ew_u.parts.msw;
(ix1) = ew_u.parts.lsw;
}

} 

} 

} 


#include <hydra/detail/external/hydra_thrust/detail/complex/c99math.h>
