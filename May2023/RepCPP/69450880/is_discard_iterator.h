

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <hydra/detail/external/hydra_thrust/iterator/discard_iterator.h>

namespace hydra_thrust
{
namespace detail
{

template <typename Iterator>
struct is_discard_iterator
: public hydra_thrust::detail::false_type
{};

template <typename System>
struct is_discard_iterator< hydra_thrust::discard_iterator<System> >
: public hydra_thrust::detail::true_type
{};

} 
} 

