#pragma once
#include <cuda_runtime_api.h>
#include <mma.h> 

#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>

using namespace nvcuda;

namespace unum {

struct cuda_base_t {
static constexpr int max_block_size_k = 1024;
static constexpr int threads = 512;

int blocks = max_block_size_k;
thrust::device_vector<float> gpu_inputs;
thrust::device_vector<float> gpu_partial_sums;
thrust::host_vector<float> cpu_partial_sums;

cuda_base_t(float const *b, float const *e)
: blocks(std::min<int>(((e - b) + threads - 1) / threads, max_block_size_k)), gpu_inputs(b, e),
gpu_partial_sums(max_block_size_k), cpu_partial_sums(max_block_size_k) {}
};

__global__ void cu_recude_blocks(const float *inputs, unsigned int input_size, float *outputs) {
extern __shared__ float shared[];
unsigned int const tid = threadIdx.x;

shared[tid] = inputs[threadIdx.x + blockDim.x * blockIdx.x];
__syncthreads();

for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
if (tid < s)
shared[tid] += shared[tid + s];
__syncthreads();
}

if (tid == 0)
outputs[blockIdx.x] = shared[0];
}


struct cuda_blocks_t : public cuda_base_t {

cuda_blocks_t(float const *b, float const *e) : cuda_base_t(b, e) {}

float operator()() {

int shared_memory = threads * sizeof(float);
cu_recude_blocks<<<blocks, threads, shared_memory>>>(gpu_inputs.data().get(), gpu_inputs.size(),
gpu_partial_sums.data().get());

shared_memory = max_block_size_k * sizeof(float);
cu_recude_blocks<<<1, max_block_size_k, shared_memory>>>(gpu_partial_sums.data().get(), blocks,
gpu_partial_sums.data().get());

cudaDeviceSynchronize();
cpu_partial_sums = gpu_partial_sums;
return cpu_partial_sums[0];
}
};

__inline__ __device__ float cu_reduce_warp(float val) {
val += __shfl_down_sync(0xffffffff, val, 16);
val += __shfl_down_sync(0xffffffff, val, 8);
val += __shfl_down_sync(0xffffffff, val, 4);
val += __shfl_down_sync(0xffffffff, val, 2);
val += __shfl_down_sync(0xffffffff, val, 1);
return val;
}

__global__ void cu_reduce_warps(float const *inputs, unsigned int input_size, float *outputs) {
float sum = 0;
for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < input_size; i += blockDim.x * gridDim.x)
sum += inputs[i];

__shared__ float shared[32];
unsigned int lane = threadIdx.x % warpSize;
unsigned int wid = threadIdx.x / warpSize;

sum = cu_reduce_warp(sum);

if (lane == 0)
shared[wid] = sum;

__syncthreads();

sum = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

if (wid == 0)
sum = cu_reduce_warp(sum);

if (threadIdx.x == 0)
outputs[blockIdx.x] = sum;
}


struct cuda_warps_t : public cuda_base_t {

cuda_warps_t(float const *b, float const *e) : cuda_base_t(b, e) {}

float operator()() {

cu_reduce_warps<<<blocks, threads>>>(gpu_inputs.data().get(), gpu_inputs.size(), gpu_partial_sums.data().get());

cu_reduce_warps<<<1, max_block_size_k>>>(gpu_partial_sums.data().get(), blocks, gpu_partial_sums.data().get());

cudaDeviceSynchronize();
cpu_partial_sums = gpu_partial_sums;
return cpu_partial_sums[0];
}
};

inline static size_t cuda_device_count() {
int count;
auto error = cudaGetDeviceCount(&count);
if (error != cudaSuccess)
return 0;
return static_cast<size_t>(count);
}


struct cuda_thrust_t {
thrust::device_vector<float> gpu_inputs;
cuda_thrust_t(float const *b, float const *e) : gpu_inputs(b, e) {}
float operator()() const {
return thrust::reduce(gpu_inputs.begin(), gpu_inputs.end(), float(0), thrust::plus<float>());
}
};


struct cuda_cub_t {
thrust::device_vector<float> gpu_inputs;
thrust::device_vector<uint8_t> temporary;
thrust::device_vector<float> gpu_sums;
thrust::host_vector<float> cpu_sums;

cuda_cub_t(float const *b, float const *e) : gpu_inputs(b, e), gpu_sums(1), cpu_sums(1) {
assert(gpu_inputs.size() < std::numeric_limits<int>::max());
}

float operator()() {

auto num_items = static_cast<int>(gpu_inputs.size());
auto d_in = gpu_inputs.data().get();
auto d_out = gpu_sums.data().get();
cudaError_t error;

void *d_temp_storage = nullptr;
size_t temp_storage_bytes = 0;
error = cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
assert(error == cudaSuccess);
assert(temp_storage_bytes > 0);

if (temp_storage_bytes > temporary.size())
temporary.resize(temp_storage_bytes);
d_temp_storage = temporary.data().get();

error = cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
assert(error == cudaSuccess);
cudaDeviceSynchronize();

cpu_sums = gpu_sums;
return cpu_sums[0];
}
};

} 