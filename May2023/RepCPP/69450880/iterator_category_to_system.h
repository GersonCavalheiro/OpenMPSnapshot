

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_categories.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/iterator_traversal_tags.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/host_system_tag.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/device_system_tag.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/any_system_tag.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>

namespace hydra_thrust
{

namespace detail
{

template <typename> struct is_iterator_system;

template <typename> struct device_iterator_category_to_backend_system;

template<typename Category>
struct iterator_category_to_system
: eval_if<
or_<
is_convertible<Category, hydra_thrust::input_host_iterator_tag>,
is_convertible<Category, hydra_thrust::output_host_iterator_tag>
>::value,

detail::identity_<hydra_thrust::host_system_tag>,

eval_if<
or_<
is_convertible<Category, hydra_thrust::input_device_iterator_tag>,
is_convertible<Category, hydra_thrust::output_device_iterator_tag>
>::value,

detail::identity_<hydra_thrust::device_system_tag>,

detail::identity_<void>
> 
> 
{
}; 


template<typename CategoryOrTraversal>
struct iterator_category_or_traversal_to_system
: eval_if<
is_iterator_system<CategoryOrTraversal>::value,
detail::identity_<CategoryOrTraversal>,
iterator_category_to_system<CategoryOrTraversal>
>
{
}; 

} 
} 

