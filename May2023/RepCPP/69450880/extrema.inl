




#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/get_iterator_value.h>
#include <hydra/detail/external/hydra_thrust/extrema.h>
#include <hydra/detail/external/hydra_thrust/functional.h>
#include <hydra/detail/external/hydra_thrust/pair.h>
#include <hydra/detail/external/hydra_thrust/reduce.h>
#include <hydra/detail/external/hydra_thrust/transform_reduce.h>

#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/iterator/counting_iterator.h>
#include <hydra/detail/external/hydra_thrust/iterator/zip_iterator.h>

namespace hydra_thrust
{
namespace system
{
namespace detail
{
namespace generic
{
namespace detail
{



template <typename InputType, typename IndexType, typename BinaryPredicate>
struct min_element_reduction
{
BinaryPredicate comp;

__host__ __device__ 
min_element_reduction(BinaryPredicate comp) : comp(comp){}

__host__ __device__ 
hydra_thrust::tuple<InputType, IndexType>
operator()(const hydra_thrust::tuple<InputType, IndexType>& lhs, 
const hydra_thrust::tuple<InputType, IndexType>& rhs )
{
if(comp(hydra_thrust::get<0>(lhs), hydra_thrust::get<0>(rhs)))
return lhs;
if(comp(hydra_thrust::get<0>(rhs), hydra_thrust::get<0>(lhs)))
return rhs;

if(hydra_thrust::get<1>(lhs) < hydra_thrust::get<1>(rhs))
return lhs;
else
return rhs;
} 
}; 


template <typename InputType, typename IndexType, typename BinaryPredicate>
struct max_element_reduction
{
BinaryPredicate comp;

__host__ __device__ 
max_element_reduction(BinaryPredicate comp) : comp(comp){}

__host__ __device__ 
hydra_thrust::tuple<InputType, IndexType>
operator()(const hydra_thrust::tuple<InputType, IndexType>& lhs, 
const hydra_thrust::tuple<InputType, IndexType>& rhs )
{
if(comp(hydra_thrust::get<0>(lhs), hydra_thrust::get<0>(rhs)))
return rhs;
if(comp(hydra_thrust::get<0>(rhs), hydra_thrust::get<0>(lhs)))
return lhs;

if(hydra_thrust::get<1>(lhs) < hydra_thrust::get<1>(rhs))
return lhs;
else
return rhs;
} 
}; 


template <typename InputType, typename IndexType, typename BinaryPredicate>
struct minmax_element_reduction
{
BinaryPredicate comp;

__host__ __device__
minmax_element_reduction(BinaryPredicate comp) : comp(comp){}

__host__ __device__ 
hydra_thrust::tuple< hydra_thrust::tuple<InputType,IndexType>, hydra_thrust::tuple<InputType,IndexType> >
operator()(const hydra_thrust::tuple< hydra_thrust::tuple<InputType,IndexType>, hydra_thrust::tuple<InputType,IndexType> >& lhs, 
const hydra_thrust::tuple< hydra_thrust::tuple<InputType,IndexType>, hydra_thrust::tuple<InputType,IndexType> >& rhs )
{

return hydra_thrust::make_tuple(min_element_reduction<InputType, IndexType, BinaryPredicate>(comp)(hydra_thrust::get<0>(lhs), hydra_thrust::get<0>(rhs)),
max_element_reduction<InputType, IndexType, BinaryPredicate>(comp)(hydra_thrust::get<1>(lhs), hydra_thrust::get<1>(rhs)));
} 
}; 


template <typename InputType, typename IndexType>
struct duplicate_tuple
{
__host__ __device__ 
hydra_thrust::tuple< hydra_thrust::tuple<InputType,IndexType>, hydra_thrust::tuple<InputType,IndexType> >
operator()(const hydra_thrust::tuple<InputType,IndexType>& t)
{
return hydra_thrust::make_tuple(t, t);
}
}; 


} 


template <typename DerivedPolicy, typename ForwardIterator>
__host__ __device__
ForwardIterator min_element(hydra_thrust::execution_policy<DerivedPolicy> &exec,
ForwardIterator first,
ForwardIterator last)
{
typedef typename hydra_thrust::iterator_value<ForwardIterator>::type value_type;

return hydra_thrust::min_element(exec, first, last, hydra_thrust::less<value_type>());
} 


template <typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
__host__ __device__
ForwardIterator min_element(hydra_thrust::execution_policy<DerivedPolicy> &exec,
ForwardIterator first,
ForwardIterator last,
BinaryPredicate comp)
{
if (first == last)
return last;

typedef typename hydra_thrust::iterator_traits<ForwardIterator>::value_type      InputType;
typedef typename hydra_thrust::iterator_traits<ForwardIterator>::difference_type IndexType;

hydra_thrust::tuple<InputType, IndexType> result =
hydra_thrust::reduce
(exec,
hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(first, hydra_thrust::counting_iterator<IndexType>(0))),
hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(first, hydra_thrust::counting_iterator<IndexType>(0))) + (last - first),
hydra_thrust::tuple<InputType, IndexType>(hydra_thrust::detail::get_iterator_value(derived_cast(exec), first), 0),
detail::min_element_reduction<InputType, IndexType, BinaryPredicate>(comp));

return first + hydra_thrust::get<1>(result);
} 


template <typename DerivedPolicy, typename ForwardIterator>
__host__ __device__
ForwardIterator max_element(hydra_thrust::execution_policy<DerivedPolicy> &exec,
ForwardIterator first,
ForwardIterator last)
{
typedef typename hydra_thrust::iterator_value<ForwardIterator>::type value_type;

return hydra_thrust::max_element(exec, first, last, hydra_thrust::less<value_type>());
} 


template <typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
__host__ __device__
ForwardIterator max_element(hydra_thrust::execution_policy<DerivedPolicy> &exec,
ForwardIterator first,
ForwardIterator last,
BinaryPredicate comp)
{
if (first == last)
return last;

typedef typename hydra_thrust::iterator_traits<ForwardIterator>::value_type      InputType;
typedef typename hydra_thrust::iterator_traits<ForwardIterator>::difference_type IndexType;

hydra_thrust::tuple<InputType, IndexType> result =
hydra_thrust::reduce
(exec,
hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(first, hydra_thrust::counting_iterator<IndexType>(0))),
hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(first, hydra_thrust::counting_iterator<IndexType>(0))) + (last - first),
hydra_thrust::tuple<InputType, IndexType>(hydra_thrust::detail::get_iterator_value(derived_cast(exec),first), 0),
detail::max_element_reduction<InputType, IndexType, BinaryPredicate>(comp));

return first + hydra_thrust::get<1>(result);
} 


template <typename DerivedPolicy, typename ForwardIterator>
__host__ __device__
hydra_thrust::pair<ForwardIterator,ForwardIterator> minmax_element(hydra_thrust::execution_policy<DerivedPolicy> &exec,
ForwardIterator first, 
ForwardIterator last)
{
typedef typename hydra_thrust::iterator_value<ForwardIterator>::type value_type;

return hydra_thrust::minmax_element(exec, first, last, hydra_thrust::less<value_type>());
} 


template <typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
__host__ __device__
hydra_thrust::pair<ForwardIterator,ForwardIterator> minmax_element(hydra_thrust::execution_policy<DerivedPolicy> &exec,
ForwardIterator first, 
ForwardIterator last,
BinaryPredicate comp)
{
if (first == last)
return hydra_thrust::make_pair(last, last);

typedef typename hydra_thrust::iterator_traits<ForwardIterator>::value_type      InputType;
typedef typename hydra_thrust::iterator_traits<ForwardIterator>::difference_type IndexType;

hydra_thrust::tuple< hydra_thrust::tuple<InputType,IndexType>, hydra_thrust::tuple<InputType,IndexType> > result = 
hydra_thrust::transform_reduce
(exec,
hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(first, hydra_thrust::counting_iterator<IndexType>(0))),
hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(first, hydra_thrust::counting_iterator<IndexType>(0))) + (last - first),
detail::duplicate_tuple<InputType, IndexType>(),
detail::duplicate_tuple<InputType, IndexType>()(
hydra_thrust::tuple<InputType, IndexType>(hydra_thrust::detail::get_iterator_value(derived_cast(exec),first), 0)),
detail::minmax_element_reduction<InputType, IndexType, BinaryPredicate>(comp));

return hydra_thrust::make_pair(first + hydra_thrust::get<1>(hydra_thrust::get<0>(result)), first + hydra_thrust::get<1>(hydra_thrust::get<1>(result)));
} 


} 
} 
} 
} 

