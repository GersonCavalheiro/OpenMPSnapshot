#pragma once
#include "setup.hpp"
#if(!(GLM_ARCH & GLM_ARCH_SSE2))
#	error "SSE2 instructions not supported or enabled"
#else
#include "intrinsic_geometric.hpp"
namespace glm{
namespace detail
{
void sse_add_ps(__m128 in1[4], __m128 in2[4], __m128 out[4]);
void sse_sub_ps(__m128 in1[4], __m128 in2[4], __m128 out[4]);
__m128 sse_mul_ps(__m128 m[4], __m128 v);
__m128 sse_mul_ps(__m128 v, __m128 m[4]);
void sse_mul_ps(__m128 const in1[4], __m128 const in2[4], __m128 out[4]);
void sse_transpose_ps(__m128 const in[4], __m128 out[4]);
void sse_inverse_ps(__m128 const in[4], __m128 out[4]);
void sse_rotate_ps(__m128 const in[4], float Angle, float const v[3], __m128 out[4]);
__m128 sse_det_ps(__m128 const m[4]);
__m128 sse_slow_det_ps(__m128 const m[4]);
}
}
#include "intrinsic_matrix.inl"
#endif
