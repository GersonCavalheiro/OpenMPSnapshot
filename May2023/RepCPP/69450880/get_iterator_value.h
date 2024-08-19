#pragma once


#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/execution_policy.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits/pointer_traits.h>
#include <hydra/detail/external/hydra_thrust/system/detail/generic/memory.h> 

namespace hydra_thrust {
namespace detail {

template<typename DerivedPolicy, typename Iterator>
__host__ __device__
typename hydra_thrust::iterator_traits<Iterator>::value_type
get_iterator_value(hydra_thrust::execution_policy<DerivedPolicy> &, Iterator it)
{
return *it;
} 

template<typename DerivedPolicy, typename Pointer>
__host__ __device__
typename hydra_thrust::detail::pointer_traits<Pointer*>::element_type 
get_iterator_value(hydra_thrust::execution_policy<DerivedPolicy> &exec, Pointer* ptr)
{
return get_value(derived_cast(exec),ptr);
} 

} 
} 
