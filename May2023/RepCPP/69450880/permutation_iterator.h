





#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/permutation_iterator_base.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_facade.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>

namespace hydra_thrust
{







template <typename ElementIterator,
typename IndexIterator>
class permutation_iterator
: public hydra_thrust::detail::permutation_iterator_base<
ElementIterator,
IndexIterator
>::type
{

private:
typedef typename detail::permutation_iterator_base<ElementIterator,IndexIterator>::type super_t;

friend class hydra_thrust::iterator_core_access;


public:

__host__ __device__
permutation_iterator()
: m_element_iterator() {}


__host__ __device__
explicit permutation_iterator(ElementIterator x, IndexIterator y)
: super_t(y), m_element_iterator(x) {}


template<typename OtherElementIterator, typename OtherIndexIterator>
__host__ __device__
permutation_iterator(permutation_iterator<OtherElementIterator,OtherIndexIterator> const &r
, typename detail::enable_if_convertible<OtherElementIterator, ElementIterator>::type* = 0
, typename detail::enable_if_convertible<OtherIndexIterator, IndexIterator>::type* = 0
)
: super_t(r.base()), m_element_iterator(r.m_element_iterator)
{}


private:
HYDRA_THRUST_DISABLE_MSVC_WARNING_BEGIN(4172)

__hydra_thrust_exec_check_disable__
__host__ __device__
typename super_t::reference dereference() const
{
return *(m_element_iterator + *this->base());
}

HYDRA_THRUST_DISABLE_MSVC_WARNING_END(4172)

template<typename,typename> friend class permutation_iterator;

ElementIterator m_element_iterator;

}; 



template<typename ElementIterator, typename IndexIterator>
__host__ __device__
permutation_iterator<ElementIterator,IndexIterator> make_permutation_iterator(ElementIterator e, IndexIterator i)
{
return permutation_iterator<ElementIterator,IndexIterator>(e,i);
}





} 

