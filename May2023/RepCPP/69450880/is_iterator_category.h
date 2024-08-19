

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_categories.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>

namespace hydra_thrust
{

namespace detail
{

template <typename T>
struct is_host_iterator_category
: hydra_thrust::detail::or_<
hydra_thrust::detail::is_convertible<T, hydra_thrust::input_host_iterator_tag>,
hydra_thrust::detail::is_convertible<T, hydra_thrust::output_host_iterator_tag>
>
{
}; 

template <typename T>
struct is_device_iterator_category
: hydra_thrust::detail::or_<
hydra_thrust::detail::is_convertible<T, hydra_thrust::input_device_iterator_tag>,
hydra_thrust::detail::is_convertible<T, hydra_thrust::output_device_iterator_tag>
>
{
}; 


template <typename T>
struct is_iterator_category
: hydra_thrust::detail::or_<
is_host_iterator_category<T>,
is_device_iterator_category<T>
>
{
}; 

} 

} 

