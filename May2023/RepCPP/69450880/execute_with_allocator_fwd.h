

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>

#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>

#if HYDRA_THRUST_CPP_DIALECT >= 2011
#include <hydra/detail/external/hydra_thrust/detail/execute_with_dependencies.h>
#endif

namespace hydra_thrust
{
namespace detail
{

template <typename Allocator, template <typename> class BaseSystem>
struct execute_with_allocator
: BaseSystem<execute_with_allocator<Allocator, BaseSystem> >
{
private:
typedef BaseSystem<execute_with_allocator<Allocator, BaseSystem> > super_t;

Allocator alloc;

public:
__host__ __device__
execute_with_allocator(super_t const& super, Allocator alloc_)
: super_t(super), alloc(alloc_)
{}

__hydra_thrust_exec_check_disable__
__host__ __device__
execute_with_allocator(Allocator alloc_)
: alloc(alloc_)
{}

typename remove_reference<Allocator>::type& get_allocator() { return alloc; }

#if HYDRA_THRUST_CPP_DIALECT >= 2011
template<typename ...Dependencies>
__host__
execute_with_allocator_and_dependencies<Allocator, BaseSystem, Dependencies...>
after(Dependencies&& ...dependencies) const
{
return { alloc, capture_as_dependency(HYDRA_THRUST_FWD(dependencies))... };
}

template<typename ...Dependencies>
__host__
execute_with_allocator_and_dependencies<Allocator, BaseSystem, Dependencies...>
after(std::tuple<Dependencies...>& dependencies) const
{
return { alloc, capture_as_dependency(dependencies) };
}
template<typename ...Dependencies>
__host__
execute_with_allocator_and_dependencies<Allocator, BaseSystem, Dependencies...>
after(std::tuple<Dependencies...>&& dependencies) const
{
return { alloc, capture_as_dependency(std::move(dependencies)) };
}

template<typename ...Dependencies>
__host__
execute_with_allocator_and_dependencies<Allocator, BaseSystem, Dependencies...>
rebind_after(Dependencies&& ...dependencies) const
{
return { alloc, capture_as_dependency(HYDRA_THRUST_FWD(dependencies))... };
}

template<typename ...Dependencies>
__host__
execute_with_allocator_and_dependencies<Allocator, BaseSystem, Dependencies...>
rebind_after(std::tuple<Dependencies...>& dependencies) const
{
return { alloc, capture_as_dependency(dependencies) };
}
template<typename ...Dependencies>
__host__
execute_with_allocator_and_dependencies<Allocator, BaseSystem, Dependencies...>
rebind_after(std::tuple<Dependencies...>&& dependencies) const
{
return { alloc, capture_as_dependency(std::move(dependencies)) };
}
#endif
};

}} 
