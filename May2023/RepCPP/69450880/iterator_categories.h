







#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/iterator_category_with_system_and_traversal.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/iterator_traversal_tags.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/device_system_tag.h>

#include <iterator>

namespace hydra_thrust
{




struct input_device_iterator_tag
: hydra_thrust::detail::iterator_category_with_system_and_traversal<
std::input_iterator_tag,
hydra_thrust::device_system_tag,
hydra_thrust::single_pass_traversal_tag
>
{};


struct output_device_iterator_tag
: hydra_thrust::detail::iterator_category_with_system_and_traversal<
std::output_iterator_tag,
hydra_thrust::device_system_tag,
hydra_thrust::single_pass_traversal_tag
>
{};


struct forward_device_iterator_tag
: hydra_thrust::detail::iterator_category_with_system_and_traversal<
std::forward_iterator_tag,
hydra_thrust::device_system_tag,
hydra_thrust::forward_traversal_tag
>
{};


struct bidirectional_device_iterator_tag
: hydra_thrust::detail::iterator_category_with_system_and_traversal<
std::bidirectional_iterator_tag,
hydra_thrust::device_system_tag,
hydra_thrust::bidirectional_traversal_tag
>
{};


struct random_access_device_iterator_tag
: hydra_thrust::detail::iterator_category_with_system_and_traversal<
std::random_access_iterator_tag,
hydra_thrust::device_system_tag,
hydra_thrust::random_access_traversal_tag
>
{};


typedef std::input_iterator_tag input_host_iterator_tag;


typedef std::output_iterator_tag output_host_iterator_tag;


typedef std::forward_iterator_tag forward_host_iterator_tag;


typedef std::bidirectional_iterator_tag bidirectional_host_iterator_tag;


typedef std::random_access_iterator_tag random_access_host_iterator_tag;



} 

#include <hydra/detail/external/hydra_thrust/iterator/detail/universal_categories.h>

