






#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/reverse_iterator_base.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_facade.h>

namespace hydra_thrust
{






template<typename BidirectionalIterator>
class reverse_iterator
: public detail::reverse_iterator_base<BidirectionalIterator>::type
{

private:
typedef typename hydra_thrust::detail::reverse_iterator_base<
BidirectionalIterator
>::type super_t;

friend class hydra_thrust::iterator_core_access;


public:

__host__ __device__
reverse_iterator() {}


__host__ __device__
explicit reverse_iterator(BidirectionalIterator x);


template<typename OtherBidirectionalIterator>
__host__ __device__
reverse_iterator(reverse_iterator<OtherBidirectionalIterator> const &r
#ifndef _MSC_VER
, typename hydra_thrust::detail::enable_if<
hydra_thrust::detail::is_convertible<
OtherBidirectionalIterator,
BidirectionalIterator
>::value
>::type * = 0
#endif 
);


private:
__hydra_thrust_exec_check_disable__
__host__ __device__
typename super_t::reference dereference() const;

__host__ __device__
void increment();

__host__ __device__
void decrement();

__host__ __device__
void advance(typename super_t::difference_type n);

template<typename OtherBidirectionalIterator>
__host__ __device__
typename super_t::difference_type
distance_to(reverse_iterator<OtherBidirectionalIterator> const &y) const;

}; 



template<typename BidirectionalIterator>
__host__ __device__
reverse_iterator<BidirectionalIterator> make_reverse_iterator(BidirectionalIterator x);






} 

#include <hydra/detail/external/hydra_thrust/iterator/detail/reverse_iterator.inl>

