

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/system/detail/sequential/stable_primitive_sort.h>
#include <hydra/detail/external/hydra_thrust/system/detail/sequential/stable_radix_sort.h>
#include <hydra/detail/external/hydra_thrust/functional.h>
#include <hydra/detail/external/hydra_thrust/system/detail/sequential/partition.h>
#include <hydra/detail/external/hydra_thrust/iterator/zip_iterator.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>

namespace hydra_thrust
{
namespace system
{
namespace detail
{
namespace sequential
{
namespace stable_primitive_sort_detail
{


template<typename Iterator>
struct enable_if_bool_sort
: hydra_thrust::detail::enable_if<
hydra_thrust::detail::is_same<
bool,
typename hydra_thrust::iterator_value<Iterator>::type
>::value
>
{};


template<typename Iterator>
struct disable_if_bool_sort
: hydra_thrust::detail::disable_if<
hydra_thrust::detail::is_same<
bool,
typename hydra_thrust::iterator_value<Iterator>::type
>::value
>
{};



template<typename DerivedPolicy,
typename RandomAccessIterator>
typename enable_if_bool_sort<RandomAccessIterator>::type
__host__ __device__
stable_primitive_sort(sequential::execution_policy<DerivedPolicy> &exec,
RandomAccessIterator first, RandomAccessIterator last)
{
sequential::stable_partition(exec, first, last, hydra_thrust::logical_not<bool>());
}


template<typename DerivedPolicy,
typename RandomAccessIterator>
typename disable_if_bool_sort<RandomAccessIterator>::type
__host__ __device__
stable_primitive_sort(sequential::execution_policy<DerivedPolicy> &exec,
RandomAccessIterator first, RandomAccessIterator last)
{
sequential::stable_radix_sort(exec,first,last);
}


struct logical_not_first
{
template<typename Tuple>
__host__ __device__
bool operator()(Tuple t)
{
return !hydra_thrust::get<0>(t);
}
};


template<typename DerivedPolicy,
typename RandomAccessIterator1,
typename RandomAccessIterator2>
typename enable_if_bool_sort<RandomAccessIterator1>::type
__host__ __device__
stable_primitive_sort_by_key(sequential::execution_policy<DerivedPolicy> &exec,
RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last,
RandomAccessIterator2 values_first)
{
sequential::stable_partition(exec,
hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(keys_first, values_first)),
hydra_thrust::make_zip_iterator(hydra_thrust::make_tuple(keys_last, values_first)),
logical_not_first());
}


template<typename DerivedPolicy,
typename RandomAccessIterator1,
typename RandomAccessIterator2>
typename disable_if_bool_sort<RandomAccessIterator1>::type
__host__ __device__
stable_primitive_sort_by_key(sequential::execution_policy<DerivedPolicy> &exec,
RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last,
RandomAccessIterator2 values_first)
{
sequential::stable_radix_sort_by_key(exec, keys_first, keys_last, values_first);
}


} 


template<typename DerivedPolicy,
typename RandomAccessIterator>
__host__ __device__
void stable_primitive_sort(sequential::execution_policy<DerivedPolicy> &exec,
RandomAccessIterator first,
RandomAccessIterator last)
{
stable_primitive_sort_detail::stable_primitive_sort(exec, first,last);
}


template<typename DerivedPolicy,
typename RandomAccessIterator1,
typename RandomAccessIterator2>
__host__ __device__
void stable_primitive_sort_by_key(sequential::execution_policy<DerivedPolicy> &exec,
RandomAccessIterator1 keys_first,
RandomAccessIterator1 keys_last,
RandomAccessIterator2 values_first)
{
stable_primitive_sort_detail::stable_primitive_sort_by_key(exec, keys_first, keys_last, values_first);
}


} 
} 
} 
} 

