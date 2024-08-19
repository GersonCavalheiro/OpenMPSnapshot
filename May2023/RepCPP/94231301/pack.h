
#pragma once

#include "config.h"

namespace psimd {

template <typename T, int W = DEFAULT_WIDTH>
struct pack
{
pack() = default;
pack(T value);

#pragma omp declare simd
const T& operator[](int i) const;
#pragma omp declare simd
T& operator[](int i);

template <typename OTHER_T>
pack<OTHER_T, W> as();


enum {static_size = W};
using type = T;


PSIMD_ALIGN(16) T data[W];
};

template <int W = DEFAULT_WIDTH>
using mask = pack<int, W>;


template <typename T, int W>
inline pack<T, W>::pack(T value)
{
#pragma omp simd
for(int i = 0; i < W; ++i)
data[i] = value;
}

template <typename T, int W>
inline const T& pack<T, W>::operator[](int i) const
{
return data[i];
}

template <typename T, int W>
inline T& pack<T, W>::operator[](int i)
{
return data[i];
}

template <typename T, int W>
template <typename OTHER_T>
inline pack<OTHER_T, W> pack<T, W>::as()
{
pack<OTHER_T, W> result;

#pragma omp simd
for (int i = 0; i < W; ++i)
result[i] = data[i];

return result;
}

} 