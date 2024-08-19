

#pragma once

#include <hydra/detail/external/hydra_thrust/iterator/iterator_adaptor.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>

namespace hydra_thrust
{

template <typename> class reverse_iterator;

namespace detail
{

template<typename BidirectionalIterator>
struct reverse_iterator_base
{
typedef hydra_thrust::iterator_adaptor<
hydra_thrust::reverse_iterator<BidirectionalIterator>,
BidirectionalIterator
> type;
}; 

} 

} 

