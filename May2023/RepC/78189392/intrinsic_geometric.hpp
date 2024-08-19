#pragma once
#include "setup.hpp"
#if(!(GLM_ARCH & GLM_ARCH_SSE2))
#	error "SSE2 instructions not supported or enabled"
#else
#include "intrinsic_common.hpp"
namespace glm{
namespace detail
{
__m128 sse_len_ps(__m128 x);
__m128 sse_dst_ps(__m128 p0, __m128 p1);
__m128 sse_dot_ps(__m128 v1, __m128 v2);
__m128 sse_dot_ss(__m128 v1, __m128 v2);
__m128 sse_xpd_ps(__m128 v1, __m128 v2);
__m128 sse_nrm_ps(__m128 v);
__m128 sse_ffd_ps(__m128 N, __m128 I, __m128 Nref);
__m128 sse_rfe_ps(__m128 I, __m128 N);
__m128 sse_rfa_ps(__m128 I, __m128 N, __m128 eta);
}
}
#include "intrinsic_geometric.inl"
#endif
