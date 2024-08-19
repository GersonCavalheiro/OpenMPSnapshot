
#pragma once

#include "../pack.h"

namespace psimd {


template <typename PACK_T>
inline PACK_T load(void* _src)
{
auto *src = (typename PACK_T::type*) _src;
PACK_T result;

#pragma omp simd
for (int i = 0; i < PACK_T::static_size; ++i)
result[i] = src[i];

return result;
}

template <typename PACK_T>
inline PACK_T load(void* _src,
const mask<PACK_T::static_size> &m)
{
auto *src = (typename PACK_T::type*) _src;
PACK_T result;

#pragma omp simd
for (int i = 0; i < PACK_T::static_size; ++i)
if (m[i])
result[i] = src[i];

return result;
}


template <typename PACK_T, typename OFFSET_T>
inline PACK_T gather(void* _src, const pack<OFFSET_T, PACK_T::static_size> &o)
{
auto *src = (typename PACK_T::type*) _src;
PACK_T result;

#pragma omp simd
for (int i = 0; i < PACK_T::static_size; ++i)
result[i] = src[o[i]];

return result;
}

template <typename PACK_T, typename OFFSET_T>
inline PACK_T gather(void* _src,
const pack<OFFSET_T, PACK_T::static_size> &o,
const mask<PACK_T::static_size> &m)
{
auto *src = (typename PACK_T::type*) _src;
PACK_T result;

#pragma omp simd
for (int i = 0; i < PACK_T::static_size; ++i)
if(m[i])
result[i] = src[o[i]];

return result;
}


template <typename PACK_T>
inline void store(const PACK_T &p, void* _dst)
{
auto *dst = (typename PACK_T::type*) _dst;

#pragma omp simd
for (int i = 0; i < PACK_T::static_size; ++i)
dst[i] = p[i];
}

template <typename PACK_T>
inline void store(const PACK_T &p,
void* _dst,
const mask<PACK_T::static_size> &m)
{
auto *dst = (typename PACK_T::type*) _dst;

#pragma omp simd
for (int i = 0; i < PACK_T::static_size; ++i)
if (m[i])
dst[i] = p[i];
}


template <typename PACK_T, typename OFFSET_T>
inline void scatter(const PACK_T &p,
void* _dst,
const pack<OFFSET_T, PACK_T::static_size> &o)
{
auto *dst = (typename PACK_T::type*) _dst;

#pragma omp simd
for (int i = 0; i < PACK_T::static_size; ++i)
dst[o[i]] = p[i];
}

template <typename PACK_T, typename OFFSET_T>
inline void scatter(const PACK_T &p,
void* _dst,
const pack<OFFSET_T, PACK_T::static_size> &o,
const mask<PACK_T::static_size> &m)
{
auto *dst = (typename PACK_T::type*) _dst;

#pragma omp simd
for (int i = 0; i < PACK_T::static_size; ++i)
if (m[i])
dst[o[i]] = p[i];
}

} 