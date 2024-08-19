#pragma once
#include "setup.hpp"
#if(!(GLM_ARCH & GLM_ARCH_SSE2))
#	error "SSE2 instructions not supported or enabled"
#else
namespace glm{
namespace detail
{
__m128 sse_abs_ps(__m128 x);
__m128 sse_sgn_ps(__m128 x);
__m128 sse_flr_ps(__m128 v);
__m128 sse_trc_ps(__m128 v);
__m128 sse_nd_ps(__m128 v);
__m128 sse_rde_ps(__m128 v);
__m128 sse_rnd_ps(__m128 x);
__m128 sse_ceil_ps(__m128 v);
__m128 sse_frc_ps(__m128 x);
__m128 sse_mod_ps(__m128 x, __m128 y);
__m128 sse_modf_ps(__m128 x, __m128i & i);
__m128 sse_clp_ps(__m128 v, __m128 minVal, __m128 maxVal);
__m128 sse_mix_ps(__m128 v1, __m128 v2, __m128 a);
__m128 sse_stp_ps(__m128 edge, __m128 x);
__m128 sse_ssp_ps(__m128 edge0, __m128 edge1, __m128 x);
__m128 sse_nan_ps(__m128 x);
__m128 sse_inf_ps(__m128 x);
}
}
#include "intrinsic_common.inl"
#endif
