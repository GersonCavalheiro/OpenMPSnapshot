

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/host_system_tag.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/device_system_tag.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/any_system_tag.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_categories.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/iterator_traversal_tags.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/is_iterator_category.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/iterator_category_with_system_and_traversal.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/iterator_category_to_traversal.h>

namespace hydra_thrust
{

namespace detail
{




template<typename System, typename Traversal, typename ValueParam, typename Reference>
struct iterator_facade_default_category;




template<typename Traversal, typename ValueParam, typename Reference>
struct iterator_facade_default_category_std :
hydra_thrust::detail::eval_if<
hydra_thrust::detail::is_convertible<Traversal, hydra_thrust::forward_traversal_tag>::value,
hydra_thrust::detail::eval_if<
hydra_thrust::detail::is_convertible<Traversal, hydra_thrust::random_access_traversal_tag>::value,
hydra_thrust::detail::identity_<std::random_access_iterator_tag>,
hydra_thrust::detail::eval_if<
hydra_thrust::detail::is_convertible<Traversal, hydra_thrust::bidirectional_traversal_tag>::value,
hydra_thrust::detail::identity_<std::bidirectional_iterator_tag>,
hydra_thrust::detail::identity_<std::forward_iterator_tag>
>
>,
hydra_thrust::detail::eval_if< 
hydra_thrust::detail::is_convertible<Traversal, hydra_thrust::single_pass_traversal_tag>::value,
hydra_thrust::detail::identity_<std::input_iterator_tag>,
hydra_thrust::detail::identity_<Traversal>
>
>
{
}; 


template<typename Traversal, typename ValueParam, typename Reference>
struct iterator_facade_default_category_host :
hydra_thrust::detail::eval_if<
hydra_thrust::detail::is_convertible<Traversal, hydra_thrust::forward_traversal_tag>::value,
hydra_thrust::detail::eval_if<
hydra_thrust::detail::is_convertible<Traversal, hydra_thrust::random_access_traversal_tag>::value,
hydra_thrust::detail::identity_<hydra_thrust::random_access_host_iterator_tag>,
hydra_thrust::detail::eval_if<
hydra_thrust::detail::is_convertible<Traversal, hydra_thrust::bidirectional_traversal_tag>::value,
hydra_thrust::detail::identity_<hydra_thrust::bidirectional_host_iterator_tag>,
hydra_thrust::detail::identity_<hydra_thrust::forward_host_iterator_tag>
>
>,
hydra_thrust::detail::eval_if< 
hydra_thrust::detail::is_convertible<Traversal, hydra_thrust::single_pass_traversal_tag>::value,
hydra_thrust::detail::identity_<hydra_thrust::input_host_iterator_tag>,
hydra_thrust::detail::identity_<Traversal>
>
>
{
}; 


template<typename Traversal, typename ValueParam, typename Reference>
struct iterator_facade_default_category_device :
hydra_thrust::detail::eval_if<
hydra_thrust::detail::is_convertible<Traversal, hydra_thrust::forward_traversal_tag>::value,
hydra_thrust::detail::eval_if<
hydra_thrust::detail::is_convertible<Traversal, hydra_thrust::random_access_traversal_tag>::value,
hydra_thrust::detail::identity_<hydra_thrust::random_access_device_iterator_tag>,
hydra_thrust::detail::eval_if<
hydra_thrust::detail::is_convertible<Traversal, hydra_thrust::bidirectional_traversal_tag>::value,
hydra_thrust::detail::identity_<hydra_thrust::bidirectional_device_iterator_tag>,
hydra_thrust::detail::identity_<hydra_thrust::forward_device_iterator_tag>
>
>,
hydra_thrust::detail::eval_if<
hydra_thrust::detail::is_convertible<Traversal, hydra_thrust::single_pass_traversal_tag>::value, 
hydra_thrust::detail::identity_<hydra_thrust::input_device_iterator_tag>,
hydra_thrust::detail::identity_<Traversal>
>
>
{
}; 


template<typename Traversal, typename ValueParam, typename Reference>
struct iterator_facade_default_category_any
{
typedef hydra_thrust::detail::iterator_category_with_system_and_traversal<
typename iterator_facade_default_category_std<Traversal, ValueParam, Reference>::type,
hydra_thrust::any_system_tag,
Traversal
> type;
}; 


template<typename System, typename Traversal, typename ValueParam, typename Reference>
struct iterator_facade_default_category
: hydra_thrust::detail::eval_if<
hydra_thrust::detail::is_convertible<System, hydra_thrust::any_system_tag>::value,
iterator_facade_default_category_any<Traversal, ValueParam, Reference>,

hydra_thrust::detail::eval_if<
hydra_thrust::detail::is_convertible<System, hydra_thrust::host_system_tag>::value,
iterator_facade_default_category_host<Traversal, ValueParam, Reference>,

hydra_thrust::detail::eval_if<
hydra_thrust::detail::is_convertible<System, hydra_thrust::device_system_tag>::value,
iterator_facade_default_category_device<Traversal, ValueParam, Reference>,

hydra_thrust::detail::identity_<
hydra_thrust::detail::iterator_category_with_system_and_traversal<
typename iterator_facade_default_category_std<Traversal, ValueParam, Reference>::type,
System,
Traversal
>
>
>
>
>
{};


template<typename System, typename Traversal, typename ValueParam, typename Reference>
struct iterator_facade_category_impl
{
typedef typename iterator_facade_default_category<
System,Traversal,ValueParam,Reference
>::type category;

typedef typename hydra_thrust::detail::eval_if<
hydra_thrust::detail::and_<
hydra_thrust::detail::is_same<
Traversal,
typename hydra_thrust::detail::iterator_category_to_traversal<category>::type
>,
hydra_thrust::detail::is_same<
System,
typename hydra_thrust::detail::iterator_category_to_system<category>::type
>
>::value,
hydra_thrust::detail::identity_<category>,
hydra_thrust::detail::identity_<hydra_thrust::detail::iterator_category_with_system_and_traversal<category,System,Traversal> >
>::type type;
}; 


template<typename CategoryOrSystem,
typename CategoryOrTraversal,
typename ValueParam,
typename Reference>
struct iterator_facade_category
{
typedef typename
hydra_thrust::detail::eval_if<
hydra_thrust::detail::is_iterator_category<CategoryOrTraversal>::value,
hydra_thrust::detail::identity_<CategoryOrTraversal>, 
iterator_facade_category_impl<CategoryOrSystem, CategoryOrTraversal, ValueParam, Reference>
>::type type;
}; 


} 
} 

