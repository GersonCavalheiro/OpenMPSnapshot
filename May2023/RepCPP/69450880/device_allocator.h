




#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/device_ptr.h>
#include <hydra/detail/external/hydra_thrust/mr/allocator.h>
#include <hydra/detail/external/hydra_thrust/memory/detail/device_system_resource.h>

#include <limits>
#include <stdexcept>

namespace hydra_thrust
{




template<typename Upstream>
class device_ptr_memory_resource HYDRA_THRUST_FINAL
: public hydra_thrust::mr::memory_resource<
device_ptr<void>
>
{
typedef typename Upstream::pointer upstream_ptr;

public:

__host__
device_ptr_memory_resource() : m_upstream(mr::get_global_resource<Upstream>())
{
}


__host__
device_ptr_memory_resource(Upstream * upstream) : m_upstream(upstream)
{
}

HYDRA_THRUST_NODISCARD __host__
virtual pointer do_allocate(std::size_t bytes, std::size_t alignment = HYDRA_THRUST_MR_DEFAULT_ALIGNMENT) HYDRA_THRUST_OVERRIDE
{
return pointer(m_upstream->do_allocate(bytes, alignment).get());
}

__host__
virtual void do_deallocate(pointer p, std::size_t bytes, std::size_t alignment) HYDRA_THRUST_OVERRIDE
{
m_upstream->do_deallocate(upstream_ptr(p.get()), bytes, alignment);
}

private:
Upstream * m_upstream;
};




template<typename T>
class device_allocator
: public hydra_thrust::mr::stateless_resource_allocator<
T,
device_ptr_memory_resource<device_memory_resource>
>
{
typedef hydra_thrust::mr::stateless_resource_allocator<
T,
device_ptr_memory_resource<device_memory_resource>
> base;

public:

template<typename U>
struct rebind
{

typedef device_allocator<U> other;
};


__host__
device_allocator() {}


__host__
device_allocator(const device_allocator& other) : base(other) {}


template<typename U>
__host__
device_allocator(const device_allocator<U>& other) : base(other) {}


__host__
~device_allocator() {}
};



} 

