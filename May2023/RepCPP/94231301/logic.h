
#pragma once

#include <type_traits>

#include "../pack.h"

namespace psimd {


template <typename T, int W>
inline mask<W> operator==(const pack<T, W> &p1, const pack<T, W> &p2)
{
mask<W> result;

#pragma omp simd
for (int i = 0; i < W; ++i)
result[i] = (p1[i] == p2[i]) ? 0xFFFFFFFF : 0x00000000;

return result;
}

template <typename T, int W, typename OTHER_T>
inline typename
std::enable_if<std::is_convertible<OTHER_T, T>::value, mask<W>>::type
operator==(const pack<T, W> &p1, const OTHER_T &v)
{
mask<W> result;

#pragma omp simd
for (int i = 0; i < W; ++i)
result[i] = (p1[i] == v) ? 0xFFFFFFFF : 0x00000000;

return result;
}

template <typename T, int W, typename OTHER_T>
inline typename
std::enable_if<std::is_convertible<OTHER_T, T>::value, mask<W>>::type
operator==(const OTHER_T &v, const pack<T, W> &p1)
{
return p1 == v;
}


template <typename T, int W>
inline mask<W> operator!=(const pack<T, W> &p1, const pack<T, W> &p2)
{
mask<W> result;

#pragma omp simd
for (int i = 0; i < W; ++i)
result[i] = (p1[i] != p2[i]) ? 0xFFFFFFFF : 0x00000000;

return result;
}

template <typename T, int W, typename OTHER_T>
inline typename
std::enable_if<std::is_convertible<OTHER_T, T>::value, mask<W>>::type
operator!=(const pack<T, W> &p1, const OTHER_T &v)
{
mask<W> result;

#pragma omp simd
for (int i = 0; i < W; ++i)
result[i] = (p1[i] != v) ? 0xFFFFFFFF : 0x00000000;

return result;
}

template <typename T, int W, typename OTHER_T>
inline typename
std::enable_if<std::is_convertible<OTHER_T, T>::value, mask<W>>::type
operator!=(const OTHER_T &v, const pack<T, W> &p1)
{
return p1 != v;
}


template <typename T, int W>
inline mask<W> operator<(const pack<T, W> &p1, const pack<T, W> &p2)
{
mask<W> result;

#pragma omp simd
for (int i = 0; i < W; ++i)
result[i] = (p1[i] < p2[i]) ? 0xFFFFFFFF : 0x00000000;

return result;
}

template <typename T, int W, typename OTHER_T>
inline typename
std::enable_if<std::is_convertible<OTHER_T, T>::value, mask<W>>::type
operator<(const pack<T, W> &p1, const OTHER_T &v)
{
mask<W> result;

#pragma omp simd
for (int i = 0; i < W; ++i)
result[i] = (p1[i] < v) ? 0xFFFFFFFF : 0x00000000;

return result;
}

template <typename T, int W, typename OTHER_T>
inline typename
std::enable_if<std::is_convertible<OTHER_T, T>::value, mask<W>>::type
operator<(const OTHER_T &v, const pack<T, W> &p1)
{
return pack<T, W>(v) < p1;
}


template <typename T, int W>
inline mask<W> operator<=(const pack<T, W> &p1, const pack<T, W> &p2)
{
mask<W> result;

#pragma omp simd
for (int i = 0; i < W; ++i)
result[i] = (p1[i] <= p2[i]) ? 0xFFFFFFFF : 0x00000000;

return result;
}

template <typename T, int W, typename OTHER_T>
inline typename
std::enable_if<std::is_convertible<OTHER_T, T>::value, mask<W>>::type
operator<=(const pack<T, W> &p1, const OTHER_T &v)
{
mask<W> result;

#pragma omp simd
for (int i = 0; i < W; ++i)
result[i] = (p1[i] <= v) ? 0xFFFFFFFF : 0x00000000;

return result;
}

template <typename T, int W, typename OTHER_T>
inline typename
std::enable_if<std::is_convertible<OTHER_T, T>::value, mask<W>>::type
operator<=(const OTHER_T &v, const pack<T, W> &p1)
{
return pack<T, W>(v) <= p1;
}


template <typename T, int W>
inline mask<W> operator>(const pack<T, W> &p1, const pack<T, W> &p2)
{
mask<W> result;

#pragma omp simd
for (int i = 0; i < W; ++i)
result[i] = (p1[i] > p2[i]) ? 0xFFFFFFFF : 0x00000000;

return result;
}

template <typename T, int W, typename OTHER_T>
inline typename
std::enable_if<std::is_convertible<OTHER_T, T>::value, mask<W>>::type
operator>(const pack<T, W> &p1, const OTHER_T &v)
{
mask<W> result;

#pragma omp simd
for (int i = 0; i < W; ++i)
result[i] = (p1[i] > v) ? 0xFFFFFFFF : 0x00000000;

return result;
}

template <typename T, int W, typename OTHER_T>
inline typename
std::enable_if<std::is_convertible<OTHER_T, T>::value, mask<W>>::type
operator>(const OTHER_T &v, const pack<T, W> &p1)
{
return pack<T, W>(v) > p1;
}


template <typename T, int W>
inline mask<W> operator>=(const pack<T, W> &p1, const pack<T, W> &p2)
{
mask<W> result;

#pragma omp simd
for (int i = 0; i < W; ++i)
result[i] = (p1[i] >= p2[i]) ? 0xFFFFFFFF : 0x00000000;

return result;
}

template <typename T, int W, typename OTHER_T>
inline typename
std::enable_if<std::is_convertible<OTHER_T, T>::value, mask<W>>::type
operator>=(const pack<T, W> &p1, const OTHER_T &v)
{
mask<W> result;

#pragma omp simd
for (int i = 0; i < W; ++i)
result[i] = (p1[i] >= v) ? 0xFFFFFFFF : 0x00000000;

return result;
}

template <typename T, int W, typename OTHER_T>
inline typename
std::enable_if<std::is_convertible<OTHER_T, T>::value, mask<W>>::type
operator>=(const OTHER_T &v, const pack<T, W> &p1)
{
return pack<T, W>(v) >= p1;
}


template <int W>
inline mask<W> operator&&(const mask<W> &m1, const mask<W> &m2)
{
mask<W> result;

#pragma omp simd
for (int i = 0; i < W; ++i)
result[i] = (m1[i] && m2[i]) ? 0xFFFFFFFF : 0x00000000;

return result;
}


template <int W>
inline mask<W> operator||(const mask<W> &m1, const mask<W> &m2)
{
mask<W> result;

#pragma omp simd
for (int i = 0; i < W; ++i)
result[i] = (m1[i] || m2[i]) ? 0xFFFFFFFF : 0x00000000;

return result;
}


template <int W>
inline mask<W> operator!(const mask<W> &m)
{
mask<W> result;

#pragma omp simd
for (int i = 0; i < W; ++i)
result[i] = !m[i] ? 0xFFFFFFFF : 0x00000000;

return result;
}

} 