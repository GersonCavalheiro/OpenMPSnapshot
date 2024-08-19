

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_adaptor.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/use_default.h>
#include <hydra/detail/external/hydra_thrust/type_traits/is_contiguous_iterator.h>

namespace hydra_thrust
{
namespace detail
{

template <typename,typename> class tagged_iterator;

template<typename Iterator, typename Tag>
struct tagged_iterator_base
{
typedef hydra_thrust::iterator_adaptor<
tagged_iterator<Iterator,Tag>,
Iterator,
typename hydra_thrust::iterator_value<Iterator>::type,
Tag,
typename hydra_thrust::iterator_traversal<Iterator>::type,
typename hydra_thrust::iterator_reference<Iterator>::type,
typename hydra_thrust::iterator_difference<Iterator>::type
> type;
}; 

template<typename Iterator, typename Tag>
class tagged_iterator
: public tagged_iterator_base<Iterator,Tag>::type
{
private:
typedef typename tagged_iterator_base<Iterator,Tag>::type super_t;

public:
__host__ __device__
tagged_iterator() {}

__host__ __device__
explicit tagged_iterator(Iterator x)
: super_t(x) {}
}; 

} 

template <typename BaseIterator, typename Tag>
struct proclaim_contiguous_iterator<
detail::tagged_iterator<BaseIterator, Tag>
> : is_contiguous_iterator<BaseIterator> {};

} 

