


#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/execution_policy.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/minimum_system.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/device_system_tag.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/any_system_tag.h>

namespace hydra_thrust
{
namespace system
{
namespace detail
{
namespace generic
{

template<typename Tag>
struct select_system1_exists;

template<typename Tag1, typename Tag2>
struct select_system2_exists;

template<typename Tag1, typename Tag2, typename Tag3>
struct select_system3_exists;

template<typename Tag1, typename Tag2, typename Tag3, typename Tag4>
struct select_system4_exists;

template<typename Tag1, typename Tag2, typename Tag3, typename Tag4, typename Tag5>
struct select_system5_exists;

template<typename Tag1, typename Tag2, typename Tag3, typename Tag4, typename Tag5, typename Tag6>
struct select_system6_exists;

template<typename System>
__host__ __device__
typename hydra_thrust::detail::disable_if<
select_system1_exists<System>::value,
System &
>::type
select_system(hydra_thrust::execution_policy<System> &system);

template<typename System1, typename System2>
__host__ __device__
typename hydra_thrust::detail::enable_if_defined<
hydra_thrust::detail::minimum_system<System1,System2>
>::type
&select_system(hydra_thrust::execution_policy<System1> &system1,
hydra_thrust::execution_policy<System2> &system2);

template<typename System1, typename System2, typename System3>
__host__ __device__
typename hydra_thrust::detail::lazy_disable_if<
select_system3_exists<System1,System2,System3>::value,
hydra_thrust::detail::minimum_system<System1,System2,System3>
>::type
&select_system(hydra_thrust::execution_policy<System1> &system1,
hydra_thrust::execution_policy<System2> &system2,
hydra_thrust::execution_policy<System3> &system3);

template<typename System1, typename System2, typename System3, typename System4>
__host__ __device__
typename hydra_thrust::detail::lazy_disable_if<
select_system4_exists<System1,System2,System3,System4>::value,
hydra_thrust::detail::minimum_system<System1,System2,System3,System4>
>::type
&select_system(hydra_thrust::execution_policy<System1> &system1,
hydra_thrust::execution_policy<System2> &system2,
hydra_thrust::execution_policy<System3> &system3,
hydra_thrust::execution_policy<System4> &system4);

template<typename System1, typename System2, typename System3, typename System4, typename System5>
__host__ __device__
typename hydra_thrust::detail::lazy_disable_if<
select_system5_exists<System1,System2,System3,System4,System5>::value,
hydra_thrust::detail::minimum_system<System1,System2,System3,System4,System5>
>::type
&select_system(hydra_thrust::execution_policy<System1> &system1,
hydra_thrust::execution_policy<System2> &system2,
hydra_thrust::execution_policy<System3> &system3,
hydra_thrust::execution_policy<System4> &system4,
hydra_thrust::execution_policy<System5> &system5);

template<typename System1, typename System2, typename System3, typename System4, typename System5, typename System6>
__host__ __device__
typename hydra_thrust::detail::lazy_disable_if<
select_system6_exists<System1,System2,System3,System4,System5,System6>::value,
hydra_thrust::detail::minimum_system<System1,System2,System3,System4,System5,System6>
>::type
&select_system(hydra_thrust::execution_policy<System1> &system1,
hydra_thrust::execution_policy<System2> &system2,
hydra_thrust::execution_policy<System3> &system3,
hydra_thrust::execution_policy<System4> &system4,
hydra_thrust::execution_policy<System5> &system5,
hydra_thrust::execution_policy<System6> &system6);

inline __host__ __device__
hydra_thrust::device_system_tag select_system(hydra_thrust::any_system_tag);

} 
} 
} 
} 

#include <hydra/detail/external/hydra_thrust/system/detail/generic/select_system.inl>
