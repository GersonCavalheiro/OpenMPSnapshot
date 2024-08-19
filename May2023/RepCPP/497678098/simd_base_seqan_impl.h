
#ifndef SEQAN_INCLUDE_SEQAN_SIMD_SIMD_BASE_SEQAN_IMPL_H_
#define SEQAN_INCLUDE_SEQAN_SIMD_SIMD_BASE_SEQAN_IMPL_H_

#include <utility>
#include <tuple>

#if defined(PLATFORM_WINDOWS_VS)

#include <intrin.h>
#elif defined(PLATFORM_GCC) && (defined(__x86_64__) || defined(__i386__))

#include <x86intrin.h>
#elif defined(SEQAN_SIMD_ENABLED)
#pragma message "You are trying to build with -DSEQAN_SIMD_ENABLED, which might be " \
"auto-defined if AVX or SSE was enabled (e.g. -march=native, -msse4, ...), " \
"but we only support x86/x86-64 architectures for SIMD vectorization! " \
"You might want to use UME::SIMD (https:
"with -DSEQAN_UMESIMD_ENABLED for a different SIMD backend."
#endif

namespace seqan {

#ifdef COMPILER_LINTEL
#include <type_traits>
#define SEQAN_VECTOR_CAST_(T, v) static_cast<typename std::decay<T>::type>(v)
#define SEQAN_VECTOR_CAST_LVALUE_(T, v) static_cast<T>(v)
#else
#define SEQAN_VECTOR_CAST_(T, v) reinterpret_cast<T>(v)
#define SEQAN_VECTOR_CAST_LVALUE_(T, v) reinterpret_cast<T>(v)
#endif



#define SEQAN_DEFINE_SIMD_VECTOR_GETVALUE_(TSimdVector)                                                 \
template <typename TPosition>                                                                           \
inline typename Value<TSimdVector>::Type                                                                \
getValue(TSimdVector & vector, TPosition const pos)                                                     \
{                                                                                                       \
return vector[pos];                                                                                 \
}

#define SEQAN_DEFINE_SIMD_VECTOR_VALUE_(TSimdVector)                                                    \
template <typename TPosition>                                                                           \
inline typename Value<TSimdVector>::Type                                                                \
value(TSimdVector & vector, TPosition const pos)                                                        \
{                                                                                                       \
return getValue(vector, pos);                                                                       \
}

#define SEQAN_DEFINE_SIMD_VECTOR_ASSIGNVALUE_(TSimdVector)                                              \
template <typename TPosition, typename TValue2>                                                         \
inline void                                                                                             \
assignValue(TSimdVector & vector, TPosition const pos, TValue2 const value)                             \
{                                                                                                       \
vector[pos] = value;                                                                                \
}

#ifdef SEQAN_SIMD_ENABLED


template <typename TValue, int LENGTH = SEQAN_SIZEOF_MAX_VECTOR / sizeof(TValue)>
struct SimdVector;

template <int VEC_SIZE, int LENGTH = 0, typename SCALAR_TYPE = void>
struct SimdParams_
{};

template <typename TSimdVector, typename TSimdParams>
struct SimdVectorTraits
{
using MaskType = TSimdVector;
};

template <int ROWS, int COLS, int BITS_PER_VALUE>
struct SimdMatrixParams_
{};

#define SEQAN_DEFINE_SIMD_VECTOR_(TSimdVector, TValue, SIZEOF_VECTOR)                                           \
typedef TValue TSimdVector __attribute__ ((__vector_size__(SIZEOF_VECTOR)));                            \
template <> struct SimdVector<TValue, SIZEOF_VECTOR / sizeof(TValue)> {  typedef TSimdVector Type; };   \
template <> struct Value<TSimdVector>           { typedef TValue Type; };                               \
template <> struct Value<TSimdVector const>:  public Value<TSimdVector> {};                             \
template <> struct LENGTH<TSimdVector>          { enum { VALUE = SIZEOF_VECTOR / sizeof(TValue) }; };   \
template <> struct LENGTH<TSimdVector const>: public LENGTH<TSimdVector> {};                            \
SEQAN_DEFINE_SIMD_VECTOR_GETVALUE_(TSimdVector)                                                         \
SEQAN_DEFINE_SIMD_VECTOR_GETVALUE_(TSimdVector const)                                                   \
SEQAN_DEFINE_SIMD_VECTOR_VALUE_(TSimdVector)                                                            \
SEQAN_DEFINE_SIMD_VECTOR_VALUE_(TSimdVector const)                                                      \
SEQAN_DEFINE_SIMD_VECTOR_ASSIGNVALUE_(TSimdVector)                                                      \
template <>                                                                                             \
SEQAN_CONCEPT_IMPL((TSimdVector),       (SimdVectorConcept));                                           \
template <>                                                                                             \
SEQAN_CONCEPT_IMPL((TSimdVector const), (SimdVectorConcept));
#endif  

} 

#endif 
