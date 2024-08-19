#pragma once

#include <complex>
#include <immintrin.h>
#include <pmmintrin.h>
#include <xmmintrin.h>

namespace simdv3 {

#define MASK(a, b, c, d) ((a << 6) + (b << 4) + (c << 2) + d)

template <typename T> struct complex2 {
T d[4];
} __attribute__((__packed__, __may_alias__));

template <> struct complex2<float> {
alignas(16) float d[4];
} __attribute__((__packed__, __may_alias__, __aligned__(16)));

template <typename T> struct complex4 {
T d[8];
} __attribute__((__packed__, __may_alias__));

template <> struct complex4<float> {
alignas(32) float d[8];
} __attribute__((__packed__, __may_alias__, __aligned__(32)));



void grid(complex2<float> &c, complex2<float> &a, complex2<float> &b) {
float *a_raw = reinterpret_cast<float(&)[4]>(a);
float *b_raw = reinterpret_cast<float(&)[4]>(b);
float *c_raw = reinterpret_cast<float(&)[4]>(c);

__m128 a_vec = _mm_loadu_ps(a_raw);
__m128 b_vec = _mm_loadu_ps(b_raw);

__m128 res_real = _mm_mul_ps(a_vec, b_vec);

__m128 twist1 = _mm_permute_ps(b_vec, MASK(2, 3, 0, 1)); 
__m128 res_imag = _mm_mul_ps(a_vec, twist1); 


__m128 mix1 = _mm_blend_ps(res_real, res_imag, 0b1010);  
__m128 mix2 = _mm_blend_ps(res_real, res_imag, 0b0101);  

mix2 = _mm_permute_ps(mix2, MASK(2, 3, 0, 1)); 

__m128 res_vec = _mm_addsub_ps(mix1, mix2);

__m128 c_vec = _mm_loadu_ps(c_raw);

c_vec = _mm_add_ps(c_vec, res_vec);

_mm_storeu_ps(c_raw, c_vec);
}

void grid(complex4<float> &c, complex4<float> &a, complex4<float> &b) {
float *a_raw = reinterpret_cast<float(&)[8]>(a);
float *b_raw = reinterpret_cast<float(&)[8]>(b);
float *c_raw = reinterpret_cast<float(&)[8]>(c);

__m256 a_vec = _mm256_loadu_ps(a_raw);
__m256 b_vec = _mm256_loadu_ps(b_raw);
__m256 c_vec = _mm256_loadu_ps(c_raw);

__m256 res_real = _mm256_mul_ps(a_vec, b_vec);

__m256 twist1 = _mm256_permute_ps(b_vec, MASK(2, 3, 0, 1)); 
__m256 res_imag = _mm256_mul_ps(a_vec, twist1); 


__m256 mix1 = _mm256_blend_ps(res_real, res_imag, 0b1010);  
__m256 mix2 = _mm256_blend_ps(res_real, res_imag, 0b0101);  

mix2 = _mm256_permute_ps(mix2, MASK(2, 3, 0, 1)); 

__m256 res_vec = _mm256_addsub_ps(mix1, mix2);

c_vec = _mm256_add_ps(c_vec, res_vec);

_mm256_storeu_ps(c_raw, c_vec);
}

} 
