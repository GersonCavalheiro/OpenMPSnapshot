
#pragma once

#include <type_traits>

#include "../pack.h"

namespace psimd {


template <typename T, int W>
inline pack<T, W> operator<<(const pack<T, W> &p1, const pack<T, W> &p2)
{
pack<T, W> result;

#pragma omp simd
for (int i = 0; i < W; ++i)
result[i] = (p1[i] << p2[i]);

return result;
}

template <typename T, int W, typename OTHER_T>
inline typename
std::enable_if<std::is_convertible<OTHER_T, T>::value, pack<T, W>>::type
operator<<(const pack<T, W> &p1, const OTHER_T &v)
{
pack<T, W> result;

#pragma omp simd
for (int i = 0; i < W; ++i)
result[i] = (p1[i] << v);

return result;
}

template <typename T, int W, typename OTHER_T>
inline typename
std::enable_if<std::is_convertible<OTHER_T, T>::value, pack<T, W>>::type
operator<<(const OTHER_T &v, const pack<T, W> &p1)
{
return pack<T, W>(v) << p1;
}


template <typename T, int W>
inline pack<T, W> operator>>(const pack<T, W> &p1, const pack<T, W> &p2)
{
pack<T, W> result;

#pragma omp simd
for (int i = 0; i < W; ++i)
result[i] = (p1[i] >> p2[i]);

return result;
}

template <typename T, int W, typename OTHER_T>
inline typename
std::enable_if<std::is_convertible<OTHER_T, T>::value, pack<T, W>>::type
operator>>(const pack<T, W> &p1, const OTHER_T &v)
{
pack<T, W> result;

#pragma omp simd
for (int i = 0; i < W; ++i)
result[i] = (p1[i] >> v);

return result;
}

template <typename T, int W, typename OTHER_T>
inline typename
std::enable_if<std::is_convertible<OTHER_T, T>::value, pack<T, W>>::type
operator>>(const OTHER_T &v, const pack<T, W> &p1)
{
return pack<T, W>(v) >> p1;
}


template <typename T, int W>
inline pack<T, W> operator^(const pack<T, W> &p1, const pack<T, W> &p2)
{
pack<T, W> result;

#pragma omp simd
for (int i = 0; i < W; ++i)
result[i] = (p1[i] ^ p2[i]);

return result;
}

template <typename T, int W, typename OTHER_T>
inline typename
std::enable_if<std::is_convertible<OTHER_T, T>::value, pack<T, W>>::type
operator^(const pack<T, W> &p1, const OTHER_T &v)
{
pack<T, W> result;

#pragma omp simd
for (int i = 0; i < W; ++i)
result[i] = (p1[i] ^ v);

return result;
}

template <typename T, int W, typename OTHER_T>
inline typename
std::enable_if<std::is_convertible<OTHER_T, T>::value, pack<T, W>>::type
operator^(const OTHER_T &v, const pack<T, W> &p1)
{
return pack<T, W>(v) ^ p1;
}

} 