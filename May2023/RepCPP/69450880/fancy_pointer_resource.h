

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/type_traits/pointer_traits.h>

#include <hydra/detail/external/hydra_thrust/mr/memory_resource.h>
#include <hydra/detail/external/hydra_thrust/mr/validator.h>

namespace hydra_thrust
{
namespace mr
{

template<typename Upstream, typename Pointer>
class fancy_pointer_resource HYDRA_THRUST_FINAL : public memory_resource<Pointer>, private validator<Upstream>
{
public:
fancy_pointer_resource() : m_upstream(get_global_resource<Upstream>())
{
}

fancy_pointer_resource(Upstream * upstream) : m_upstream(upstream)
{
}

HYDRA_THRUST_NODISCARD
virtual Pointer do_allocate(std::size_t bytes, std::size_t alignment = HYDRA_THRUST_MR_DEFAULT_ALIGNMENT) HYDRA_THRUST_OVERRIDE
{
return static_cast<Pointer>(m_upstream->do_allocate(bytes, alignment));
}

virtual void do_deallocate(Pointer p, std::size_t bytes, std::size_t alignment) HYDRA_THRUST_OVERRIDE
{
return m_upstream->do_deallocate(
static_cast<typename Upstream::pointer>(
hydra_thrust::detail::pointer_traits<Pointer>::get(p)),
bytes, alignment);
}

private:
Upstream * m_upstream;
};

} 
} 

