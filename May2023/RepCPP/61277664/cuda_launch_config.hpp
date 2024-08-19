



#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>


inline __host__ __device__
std::size_t block_size_with_maximum_potential_occupancy(const cudaFuncAttributes &attributes,
const cudaDeviceProp &properties);


template<typename UnaryFunction>
inline __host__ __device__
std::size_t block_size_with_maximum_potential_occupancy(const cudaFuncAttributes &attributes,
const cudaDeviceProp &properties,
UnaryFunction block_size_to_dynamic_smem_size);



template<typename T>
inline __host__
std::size_t maximum_residency(T t, const size_t CTA_SIZE, const size_t dynamic_smem_bytes);


inline __host__
std::size_t maximum_residency(const cudaFuncAttributes &attributes, const cudaDeviceProp &properties, 
size_t CTA_SIZE, size_t dynamic_smem_bytes);


namespace __cuda_launch_config_detail
{

using std::size_t;

namespace util
{


template<typename T>
inline __host__ __device__
T min_(const T &lhs, const T &rhs)
{
return rhs < lhs ? rhs : lhs;
}


template <typename T>
struct zero_function
{
inline __host__ __device__
T operator()(T)
{
return 0;
}
};


template<typename L, typename R>
inline __host__ __device__ L divide_ri(const L x, const R y)
{
return (x + (y - 1)) / y;
}

template<typename L, typename R>
inline __host__ __device__ L divide_rz(const L x, const R y)
{
return x / y;
}

template<typename L, typename R>
inline __host__ __device__ L round_i(const L x, const R y){ return y * divide_ri(x, y); }

template<typename L, typename R>
inline __host__ __device__ L round_z(const L x, const R y){ return y * divide_rz(x, y); }

} 



inline __host__ __device__
size_t smem_allocation_unit(const cudaDeviceProp &properties)
{
switch(properties.major)
{
case 1:  return 512;
case 2:  return 128;
case 3:  return 256;
default: return 256; 
}
}


inline __host__ __device__
size_t reg_allocation_unit(const cudaDeviceProp &properties, const size_t regsPerThread)
{
switch(properties.major)
{
case 1:  return (properties.minor <= 1) ? 256 : 512;
case 2:  switch(regsPerThread)
{
case 21:
case 22:
case 29:
case 30:
case 37:
case 38:
case 45:
case 46:
return 128;
default:
return 64;
}
case 3:  return 256;
default: return 256; 
}
}


inline __host__ __device__
size_t warp_allocation_multiple(const cudaDeviceProp &properties)
{
return (properties.major <= 1) ? 2 : 1;
}

inline __host__ __device__
size_t num_sides_per_multiprocessor(const cudaDeviceProp &properties)
{
switch(properties.major)
{
case 1:  return 1;
case 2:  return 2;
case 3:  return 4;
default: return 4; 
}
}


inline __host__ __device__
size_t max_blocks_per_multiprocessor(const cudaDeviceProp &properties)
{
return (properties.major <= 2) ? 8 : 16;
}


inline __host__ __device__
size_t max_active_blocks_per_multiprocessor(const cudaDeviceProp     &properties,
const cudaFuncAttributes &attributes,
size_t CTA_SIZE,
size_t dynamic_smem_bytes)
{

const size_t maxThreadsPerSM = properties.maxThreadsPerMultiProcessor;  
const size_t maxBlocksPerSM  = max_blocks_per_multiprocessor(properties);

const size_t ctaLimitThreads = (CTA_SIZE <= properties.maxThreadsPerBlock) ? maxThreadsPerSM / CTA_SIZE : 0;
const size_t ctaLimitBlocks  = maxBlocksPerSM;

const size_t smemAllocationUnit     = smem_allocation_unit(properties);
const size_t smemBytes  = attributes.sharedSizeBytes + dynamic_smem_bytes;
const size_t smemPerCTA = util::round_i(smemBytes, smemAllocationUnit);

const size_t ctaLimitSMem = smemPerCTA > 0 ? properties.sharedMemPerBlock / smemPerCTA : maxBlocksPerSM;

const size_t regAllocationUnit      = reg_allocation_unit(properties, attributes.numRegs);
const size_t warpAllocationMultiple = warp_allocation_multiple(properties);
const size_t numWarps = util::round_i(util::divide_ri(CTA_SIZE, properties.warpSize), warpAllocationMultiple);

size_t ctaLimitRegs;
if(properties.major <= 1)
{
const size_t regsPerCTA = util::round_i(attributes.numRegs * properties.warpSize * numWarps, regAllocationUnit);
ctaLimitRegs = regsPerCTA > 0 ? properties.regsPerBlock / regsPerCTA : maxBlocksPerSM;
}
else
{
const size_t regsPerWarp = util::round_i(attributes.numRegs * properties.warpSize, regAllocationUnit);
const size_t numSides = num_sides_per_multiprocessor(properties);
const size_t numRegsPerSide = properties.regsPerBlock / numSides;
ctaLimitRegs = regsPerWarp > 0 ? ((numRegsPerSide / regsPerWarp) * numSides) / numWarps : maxBlocksPerSM;
}

return util::min_(ctaLimitRegs, util::min_(ctaLimitSMem, util::min_(ctaLimitThreads, ctaLimitBlocks)));
}


template <typename UnaryFunction>
inline __host__ __device__
size_t default_block_size(const cudaDeviceProp     &properties,
const cudaFuncAttributes &attributes,
UnaryFunction block_size_to_smem_size)
{
size_t max_occupancy      = properties.maxThreadsPerMultiProcessor;
size_t largest_blocksize  = util::min_(properties.maxThreadsPerBlock, attributes.maxThreadsPerBlock);
size_t granularity        = properties.warpSize;
size_t max_blocksize      = 0;
size_t highest_occupancy  = 0;

for(size_t blocksize = largest_blocksize; blocksize != 0; blocksize -= granularity)
{
size_t occupancy = blocksize * max_active_blocks_per_multiprocessor(properties, attributes, blocksize, block_size_to_smem_size(blocksize));

if(occupancy > highest_occupancy)
{
max_blocksize = blocksize;
highest_occupancy = occupancy;
}

if(highest_occupancy == max_occupancy)
break;
}

return max_blocksize;
}


} 


template<typename UnaryFunction>
inline __host__ __device__
std::size_t block_size_with_maximum_potential_occupancy(const cudaFuncAttributes &attributes,
const cudaDeviceProp &properties,
UnaryFunction block_size_to_dynamic_smem_size)
{
return __cuda_launch_config_detail::default_block_size(properties, attributes, block_size_to_dynamic_smem_size);
}


inline __host__ __device__
std::size_t block_size_with_maximum_potential_occupancy(const cudaFuncAttributes &attributes,
const cudaDeviceProp &properties)
{
return block_size_with_maximum_potential_occupancy(attributes, properties, __cuda_launch_config_detail::util::zero_function<std::size_t>());
}

template<typename T>
inline __host__
std::size_t block_size_with_maximum_potential_occupancy(T t)
{
cudaError_t err;
cudaFuncAttributes attributes;
err = cudaFuncGetAttributes(&attributes, t);

if (err != cudaSuccess)
return 0;

int device;
err = cudaGetDevice(&device);

if (err != cudaSuccess)
return 0;

cudaDeviceProp properties;
err = cudaGetDeviceProperties(&properties, device);

if (err != cudaSuccess)
return 0;

return block_size_with_maximum_potential_occupancy(attributes, properties);
}

inline __host__
std::size_t maximum_residency(const cudaFuncAttributes &attributes, const cudaDeviceProp &properties,
size_t CTA_SIZE, size_t dynamic_smem_bytes)
{
return __cuda_launch_config_detail::max_active_blocks_per_multiprocessor(properties, attributes, CTA_SIZE, dynamic_smem_bytes);
}

template<typename T>
inline __host__
std::size_t maximum_residency(T t, size_t CTA_SIZE, size_t dynamic_smem_bytes)
{
cudaError_t err;
cudaFuncAttributes attributes;
err = cudaFuncGetAttributes(&attributes, t);

if (err != cudaSuccess)
return 0;

int device;
err = cudaGetDevice(&device);

if (err != cudaSuccess)
return 0;

cudaDeviceProp properties;
err = cudaGetDeviceProperties(&properties, device);

if (err != cudaSuccess)
return 0;

return maximum_residency(attributes, properties, CTA_SIZE, dynamic_smem_bytes);
}
