

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/cpp11_required.h>

#if HYDRA_THRUST_CPP_DIALECT >= 2011

#include <tuple>

#include <hydra/detail/external/hydra_thrust/detail/execute_with_dependencies.h>

namespace hydra_thrust
{
namespace detail
{

template<template<typename> class ExecutionPolicyCRTPBase>
struct dependencies_aware_execution_policy
{
template<typename ...Dependencies>
__host__
hydra_thrust::detail::execute_with_dependencies<
ExecutionPolicyCRTPBase,
Dependencies...
>
after(Dependencies&& ...dependencies) const
{
return { capture_as_dependency(HYDRA_THRUST_FWD(dependencies))... };
}

template<typename ...Dependencies>
__host__
hydra_thrust::detail::execute_with_dependencies<
ExecutionPolicyCRTPBase,
Dependencies...
>
after(std::tuple<Dependencies...>& dependencies) const
{
return { capture_as_dependency(dependencies) };
}
template<typename ...Dependencies>
__host__
hydra_thrust::detail::execute_with_dependencies<
ExecutionPolicyCRTPBase,
Dependencies...
>
after(std::tuple<Dependencies...>&& dependencies) const
{
return { capture_as_dependency(std::move(dependencies)) };
}

template<typename ...Dependencies>
__host__
hydra_thrust::detail::execute_with_dependencies<
ExecutionPolicyCRTPBase,
Dependencies...
>
rebind_after(Dependencies&& ...dependencies) const
{
return { capture_as_dependency(HYDRA_THRUST_FWD(dependencies))... };
}

template<typename ...Dependencies>
__host__
hydra_thrust::detail::execute_with_dependencies<
ExecutionPolicyCRTPBase,
Dependencies...
>
rebind_after(std::tuple<Dependencies...>& dependencies) const
{
return { capture_as_dependency(dependencies) };
}
template<typename ...Dependencies>
__host__
hydra_thrust::detail::execute_with_dependencies<
ExecutionPolicyCRTPBase,
Dependencies...
>
rebind_after(std::tuple<Dependencies...>&& dependencies) const
{
return { capture_as_dependency(std::move(dependencies)) };
}
};

} 
} 

#endif 

