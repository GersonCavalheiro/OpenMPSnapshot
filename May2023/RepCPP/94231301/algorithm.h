
#pragma once

#include "../pack.h"

namespace psimd {

template <typename T, int W, typename FCN_T>
inline void foreach(pack<T, W> &p, FCN_T &&fcn)
{
#pragma omp simd
for (int i = 0; i < W; ++i)
fcn(p[i], i);
}

template <int W, typename FCN_T>
inline void foreach_active(const mask<W> &m, FCN_T &&fcn)
{
#pragma omp simd
for (int i = 0; i < W; ++i)
if (m[i])
fcn(i);
}

template <typename T, int W, typename FCN_T>
inline void foreach_active(const mask<W> &m, pack<T, W> &p, FCN_T &&fcn)
{
#pragma omp simd
for (int i = 0; i < W; ++i)
if (m[i])
fcn(p[i]);
}

template <int W>
inline bool any(const mask<W> &m)
{
bool result = false;

#pragma omp simd
for (int i = 0; i < W; ++i)
if (m[i])
result = true;

return result;
}

template <int W>
inline bool none(const mask<W> &m)
{
return !any(m);
}

template <int W>
inline bool all(const mask<W> &m)
{
bool result = true;

#pragma omp simd
for (int i = 0; i < W; ++i)
if (!m[i])
result = false;

return result;
}

template <typename T, int W>
inline pack<T, W> select(const mask<W> &m,
const pack<T, W> &t,
const pack<T, W> &f)
{
pack<T, W> result;

#pragma omp simd
for (int i = 0; i < W; ++i) {
if (m[i])
result[i] = t[i];
else
result[i] = f[i];
}

return result;
}

} 