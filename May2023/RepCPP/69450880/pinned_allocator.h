



#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/system/cuda/detail/guarded_cuda_runtime_api.h>
#include <stdexcept>
#include <limits>
#include <string>
#include <hydra/detail/external/hydra_thrust/system/system_error.h>
#include <hydra/detail/external/hydra_thrust/system/cuda/error.h>

namespace hydra_thrust
{

namespace system
{

namespace cuda
{

namespace experimental
{




template<typename T> class pinned_allocator;

template<>
class pinned_allocator<void>
{
public:
typedef void           value_type;
typedef void       *   pointer;
typedef const void *   const_pointer;
typedef std::size_t    size_type;
typedef std::ptrdiff_t difference_type;

template<typename U>
struct rebind
{
typedef pinned_allocator<U> other;
}; 
}; 


template<typename T>
class pinned_allocator
{
public:
typedef T              value_type;
typedef T*             pointer;
typedef const T*       const_pointer;
typedef T&             reference;
typedef const T&       const_reference;
typedef std::size_t    size_type;
typedef std::ptrdiff_t difference_type;

template<typename U>
struct rebind
{
typedef pinned_allocator<U> other;
}; 


__host__ __device__
inline pinned_allocator() {}


__host__ __device__
inline ~pinned_allocator() {}


__host__ __device__
inline pinned_allocator(pinned_allocator const &) {}


template<typename U>
__host__ __device__
inline pinned_allocator(pinned_allocator<U> const &) {}


__host__ __device__
inline pointer address(reference r) { return &r; }


__host__ __device__
inline const_pointer address(const_reference r) { return &r; }


__host__
inline pointer allocate(size_type cnt,
const_pointer = 0)
{
if(cnt > this->max_size())
{
throw std::bad_alloc();
} 

pointer result(0);
cudaError_t error = cudaMallocHost(reinterpret_cast<void**>(&result), cnt * sizeof(value_type));

if(error)
{
cudaGetLastError(); 
throw std::bad_alloc();
} 

return result;
} 


__host__
inline void deallocate(pointer p, size_type )
{
cudaError_t error = cudaFreeHost(p);

cudaGetLastError(); 

if(error)
{
cudaGetLastError(); 
throw hydra_thrust::system_error(error, hydra_thrust::cuda_category());
} 
} 


inline size_type max_size() const
{
return (std::numeric_limits<size_type>::max)() / sizeof(T);
} 


__host__ __device__
inline bool operator==(pinned_allocator const& x) const { return true; }


__host__ __device__
inline bool operator!=(pinned_allocator const &x) const { return !operator==(x); }
}; 



} 

} 

} 

namespace cuda
{

namespace experimental
{

using hydra_thrust::system::cuda::experimental::pinned_allocator;

} 

} 

} 

