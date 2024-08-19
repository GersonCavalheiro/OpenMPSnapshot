

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits/function_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits/iterator/is_output_iterator.h>

namespace hydra_thrust
{

namespace detail
{


template<typename InputIterator, typename OutputIterator, typename Function>
struct intermediate_type_from_function_and_iterators
: eval_if<
has_result_type<Function>::value,
result_type<Function>,
eval_if<
is_output_iterator<OutputIterator>::value,
hydra_thrust::iterator_value<InputIterator>,
hydra_thrust::iterator_value<OutputIterator>
>
>
{
}; 

} 

} 

