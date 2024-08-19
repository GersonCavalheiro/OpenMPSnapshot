#pragma once

#include <algorithm>
#include "vector.hpp"

namespace backends
{
namespace kernels
{
template<typename T>
__global__ void fill(T *element, T value, size_t N)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if(idx < N)
{
element[idx] = value;
}
}
}

template<typename T>
void fill(vector<T>& vec, const T& val)
{
int N = vec.size();
kernels::fill<<<(n+1023)/1024, 1024>>>(vec.data(), val, N);
cudaDeviceSynchronize();
}
}