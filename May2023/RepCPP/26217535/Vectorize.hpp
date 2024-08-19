

#pragma once

#include <alpaka/core/Common.hpp>

#include <cstddef>
#include <cstdint>



namespace alpaka::core::vectorization
{
constexpr std::size_t defaultAlignment =
#if defined(__AVX512BW__) || defined(__AVX512F__) || defined(__MIC__)
64u
#elif defined(__AVX__) || defined(__AVX2__)
32u
#else
16u
#endif
;

template<typename TElem>
struct GetVectorizationSizeElems
{
static constexpr std::size_t value = 1u;
};

template<>
struct GetVectorizationSizeElems<double>
{
static constexpr std::size_t value =
#if defined(__AVX512F__) || defined(__MIC__)
8u;
#elif defined(__AVX__)
4u;
#elif defined(__SSE2__)
2u;
#elif defined(__ARM_NEON__)
1u;
#elif defined(__ALTIVEC__)
2u;
#else
1u;
#endif
};
template<>
struct GetVectorizationSizeElems<float>
{
static constexpr std::size_t value =
#if defined(__AVX512F__) || defined(__MIC__)
16u;
#elif defined(__AVX__)
8u;
#elif defined(__SSE__)
4u;
#elif defined(__ARM_NEON__)
4u;
#elif defined(__ALTIVEC__)
4u;
#else
1u;
#endif
};
template<>
struct GetVectorizationSizeElems<std::int8_t>
{
static constexpr std::size_t value =
#if defined(__AVX512BW__)
64u;
#elif defined(__AVX2__)
32u;
#elif defined(__SSE2__)
16u;
#elif defined(__ARM_NEON__)
16u;
#elif defined(__ALTIVEC__)
16u;
#elif defined(__CUDA_ARCH__)
4u;
#else
1u;
#endif
};
template<>
struct GetVectorizationSizeElems<std::uint8_t>
{
static constexpr std::size_t value =
#if defined(__AVX512BW__)
64u;
#elif defined(__AVX2__)
32u;
#elif defined(__SSE2__)
16u;
#elif defined(__ARM_NEON__)
16u;
#elif defined(__ALTIVEC__)
16u;
#elif defined(__CUDA_ARCH__)
4u;
#else
1u;
#endif
};
template<>
struct GetVectorizationSizeElems<std::int16_t>
{
static constexpr std::size_t value =
#if defined(__AVX512BW__)
32u;
#elif defined(__AVX2__)
16u;
#elif defined(__SSE2__)
8u;
#elif defined(__ARM_NEON__)
8u;
#elif defined(__ALTIVEC__)
8u;
#elif defined(__CUDA_ARCH__)
2u;
#else
1u;
#endif
};
template<>
struct GetVectorizationSizeElems<std::uint16_t>
{
static constexpr std::size_t value =
#if defined(__AVX512BW__)
32u;
#elif defined(__AVX2__)
16u;
#elif defined(__SSE2__)
8u;
#elif defined(__ARM_NEON__)
8u;
#elif defined(__ALTIVEC__)
8u;
#elif defined(__CUDA_ARCH__)
2u;
#else
1u;
#endif
};
template<>
struct GetVectorizationSizeElems<std::int32_t>
{
static constexpr std::size_t value =
#if defined(__AVX512F__) || defined(__MIC__)
16u;
#elif defined(__AVX2__)
8u;
#elif defined(__SSE2__)
4u;
#elif defined(__ARM_NEON__)
4u;
#elif defined(__ALTIVEC__)
4u;
#else
1u;
#endif
};
template<>
struct GetVectorizationSizeElems<std::uint32_t>
{
static constexpr std::size_t value =
#if defined(__AVX512F__) || defined(__MIC__)
16u;
#elif defined(__AVX2__)
8u;
#elif defined(__SSE2__)
4u;
#elif defined(__ARM_NEON__)
4u;
#elif defined(__ALTIVEC__)
4u;
#else
1u;
#endif
};
template<>
struct GetVectorizationSizeElems<std::int64_t>
{
static constexpr std::size_t value =
#if defined(__AVX512F__)
8u;
#elif defined(__AVX2__)
4u;
#elif defined(__SSE2__)
2u;
#elif defined(__ARM_NEON__)
2u;
#else
1u;
#endif
};
template<>
struct GetVectorizationSizeElems<std::uint64_t>
{
static constexpr std::size_t value =
#if defined(__AVX512F__)
8u;
#elif defined(__AVX2__)
4u;
#elif defined(__SSE2__)
2u;
#elif defined(__ARM_NEON__)
2u;
#else
1u;
#endif
};
} 
