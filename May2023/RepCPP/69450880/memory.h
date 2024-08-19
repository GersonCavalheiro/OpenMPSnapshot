



#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/system/tbb/memory_resource.h>
#include <hydra/detail/external/hydra_thrust/memory.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <hydra/detail/external/hydra_thrust/mr/allocator.h>
#include <ostream>

namespace hydra_thrust
{
namespace system
{
namespace tbb
{


inline pointer<void> malloc(std::size_t n);


template<typename T>
inline pointer<T> malloc(std::size_t n);


inline void free(pointer<void> ptr);



template<typename T>
struct allocator
: hydra_thrust::mr::stateless_resource_allocator<
T,
memory_resource
>
{
private:
typedef hydra_thrust::mr::stateless_resource_allocator<
T,
memory_resource
> base;

public:

template<typename U>
struct rebind
{

typedef allocator<U> other;
};


__host__ __device__
inline allocator() {}


__host__ __device__
inline allocator(const allocator & other) : base(other) {}


template<typename U>
__host__ __device__
inline allocator(const allocator<U> & other) : base(other) {}


__host__ __device__
inline ~allocator() {}
}; 

} 



} 


namespace tbb
{

using hydra_thrust::system::tbb::malloc;
using hydra_thrust::system::tbb::free;
using hydra_thrust::system::tbb::allocator;

} 

} 

#include <hydra/detail/external/hydra_thrust/system/tbb/detail/memory.inl>

