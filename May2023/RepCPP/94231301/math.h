
#pragma once

#include <algorithm>
#include <cmath>

#include "../pack.h"

namespace psimd {

template <typename T, int W>
inline pack<T, W> abs(const pack<T, W> &p)
{
pack<T, W> result;

#pragma omp simd
for (int i = 0; i < W; ++i)
result[i] = std::abs(p[i]);

return result;
}

template <typename T, int W>
inline pack<T, W> sqrt(const pack<T, W> &p)
{
pack<T, W> result;

#pragma omp simd
for (int i = 0; i < W; ++i)
result[i] = std::sqrt(p[i]);

return result;
}

template <typename T, int W>
inline pack<T, W> sin(const pack<T, W> &p)
{
pack<T, W> result;

#pragma omp simd
for (int i = 0; i < W; ++i)
result[i] = std::sin(p[i]);

return result;
}

template <typename T, int W>
inline pack<T, W> cos(const pack<T, W> &p)
{
pack<T, W> result;

#pragma omp simd
for (int i = 0; i < W; ++i)
result[i] = std::cos(p[i]);

return result;
}

template <typename T, int W>
inline pack<T, W> tan(const pack<T, W> &p)
{
pack<T, W> result;

#pragma omp simd
for (int i = 0; i < W; ++i)
result[i] = std::tan(p[i]);

return result;
}

template <typename T, int W>
inline pack<T, W> pow(const pack<T, W> &v, const float b)
{
pack<T, W> result;

#pragma omp simd
for (int i = 0; i < W; ++i)
result[i] = std::pow(v[i], b);

return result;
}

template <typename T, int W>
inline pack<T, W> max(const pack<T, W> &a, const pack<T, W> &b)
{
pack<T, W> result;

#pragma omp simd
for (int i = 0; i < W; ++i)
result[i] = std::max(a[i], b[i]);

return result;
}

template <typename T, int W>
inline pack<T, W> min(const pack<T, W> &a, const pack<T, W> &b)
{
pack<T, W> result;

#pragma omp simd
for (int i = 0; i < W; ++i)
result[i] = std::min(a[i], b[i]);

return result;
}

} 