

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_categories.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/iterator_traversal_tags.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/iterator_category_to_system.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>

namespace hydra_thrust
{

namespace detail
{

template <typename> struct is_iterator_system;
template <typename> struct is_iterator_traversal;

using namespace hydra_thrust::detail;

template <typename Category>
struct host_system_category_to_traversal
: eval_if<
is_convertible<Category, random_access_host_iterator_tag>::value,
detail::identity_<random_access_traversal_tag>,
eval_if<
is_convertible<Category, bidirectional_host_iterator_tag>::value,
detail::identity_<bidirectional_traversal_tag>,
eval_if<
is_convertible<Category, forward_host_iterator_tag>::value,
detail::identity_<forward_traversal_tag>,
eval_if<
is_convertible<Category, input_host_iterator_tag>::value,
detail::identity_<single_pass_traversal_tag>,
eval_if<
is_convertible<Category, output_host_iterator_tag>::value,
detail::identity_<incrementable_traversal_tag>,
void
>
>
>
>
>
{
}; 



template <typename Category>
struct device_system_category_to_traversal
: eval_if<
is_convertible<Category, random_access_device_iterator_tag>::value,
detail::identity_<random_access_traversal_tag>,
eval_if<
is_convertible<Category, bidirectional_device_iterator_tag>::value,
detail::identity_<bidirectional_traversal_tag>,
eval_if<
is_convertible<Category, forward_device_iterator_tag>::value,
detail::identity_<forward_traversal_tag>,
eval_if<
is_convertible<Category, input_device_iterator_tag>::value,
detail::identity_<single_pass_traversal_tag>,
eval_if<
is_convertible<Category, output_device_iterator_tag>::value,
detail::identity_<incrementable_traversal_tag>,
void
>
>
>
>
>
{
}; 


template<typename Category>
struct category_to_traversal
: eval_if<
or_<
is_convertible<Category, hydra_thrust::input_host_iterator_tag>,
is_convertible<Category, hydra_thrust::output_host_iterator_tag>
>::value,

host_system_category_to_traversal<Category>,

eval_if<
or_<
is_convertible<Category, hydra_thrust::input_device_iterator_tag>,
is_convertible<Category, hydra_thrust::output_device_iterator_tag>
>::value,

device_system_category_to_traversal<Category>,

void
>
>
{};


template <typename CategoryOrTraversal>
struct iterator_category_to_traversal
: eval_if<
is_iterator_traversal<CategoryOrTraversal>::value,
detail::identity_<CategoryOrTraversal>,
category_to_traversal<CategoryOrTraversal>
>
{
}; 


} 

} 

