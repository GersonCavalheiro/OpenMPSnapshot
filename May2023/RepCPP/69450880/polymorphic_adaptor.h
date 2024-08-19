

#pragma once

#include "memory_resource.h"

namespace hydra_thrust
{
namespace mr
{

template<typename Pointer = void *>
class polymorphic_adaptor_resource HYDRA_THRUST_FINAL : public memory_resource<Pointer>
{
public:
polymorphic_adaptor_resource(memory_resource<Pointer> * t) : upstream_resource(t)
{
}

virtual Pointer do_allocate(std::size_t bytes, std::size_t alignment = HYDRA_THRUST_MR_DEFAULT_ALIGNMENT) HYDRA_THRUST_OVERRIDE
{
return upstream_resource->allocate(bytes, alignment);
}

virtual void do_deallocate(Pointer p, std::size_t bytes, std::size_t alignment) HYDRA_THRUST_OVERRIDE
{
return upstream_resource->deallocate(p, bytes, alignment);
}

__host__ __device__
virtual bool do_is_equal(const memory_resource<Pointer> & other) const HYDRA_THRUST_NOEXCEPT HYDRA_THRUST_OVERRIDE
{
return upstream_resource->is_equal(other);
}

private:
memory_resource<Pointer> * upstream_resource;
};

} 
} 

