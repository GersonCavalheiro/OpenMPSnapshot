

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/execute_with_allocator_fwd.h>
#include <hydra/detail/external/hydra_thrust/detail/alignment.h>

#if HYDRA_THRUST_CPP_DIALECT >= 2011
#include <type_traits>
#endif

namespace hydra_thrust
{

namespace mr
{

template<typename T, class MR>
class allocator;

}

namespace detail
{

template<template <typename> class ExecutionPolicyCRTPBase>
struct allocator_aware_execution_policy
{
template<typename MemoryResource>
struct execute_with_memory_resource_type
{
typedef hydra_thrust::detail::execute_with_allocator<
hydra_thrust::mr::allocator<
hydra_thrust::detail::max_align_t,
MemoryResource
>,
ExecutionPolicyCRTPBase
> type;
};

template<typename Allocator>
struct execute_with_allocator_type
{
typedef hydra_thrust::detail::execute_with_allocator<
Allocator,
ExecutionPolicyCRTPBase
> type;
};

template<typename MemoryResource>
typename execute_with_memory_resource_type<MemoryResource>::type
operator()(MemoryResource * mem_res) const
{
return typename execute_with_memory_resource_type<MemoryResource>::type(mem_res);
}

template<typename Allocator>
typename execute_with_allocator_type<Allocator&>::type
operator()(Allocator &alloc) const
{
return typename execute_with_allocator_type<Allocator&>::type(alloc);
}

template<typename Allocator>
typename execute_with_allocator_type<Allocator>::type
operator()(const Allocator &alloc) const
{
return typename execute_with_allocator_type<Allocator>::type(alloc);
}

#if __cplusplus >= 201103L
template<typename Allocator,
typename std::enable_if<!std::is_lvalue_reference<Allocator>::value>::type * = nullptr>
typename execute_with_allocator_type<Allocator>::type
operator()(Allocator &&alloc) const
{
return typename execute_with_allocator_type<Allocator>::type(std::move(alloc));
}
#endif
};

}
}
