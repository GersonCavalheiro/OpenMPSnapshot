#pragma once
#include "glm/glm.hpp"
#if(!(GLM_ARCH & GLM_ARCH_SSE2))
#	error "SSE2 instructions not supported or enabled"
#else
namespace glm{
namespace detail
{
__m128i _mm_bit_interleave_si128(__m128i x);
__m128i _mm_bit_interleave_si128(__m128i x, __m128i y);
}
}
#include "intrinsic_integer.inl"
#endif
