

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/minimum_system.h>
#include <hydra/detail/external/hydra_thrust/detail/copy.h>
#include <hydra/detail/external/hydra_thrust/detail/temporary_array.h>
#include <hydra/detail/external/hydra_thrust/system/cpp/detail/execution_policy.h>

namespace hydra_thrust
{
namespace detail
{


template<typename InputIterator,
typename OutputIterator>
OutputIterator sequential_copy(InputIterator first,
InputIterator last,
OutputIterator result)
{
for(; first != last; ++first, ++result)
{
*result = *first;
} 

return result;
} 


template<typename BidirectionalIterator1,
typename BidirectionalIterator2>
BidirectionalIterator2 sequential_copy_backward(BidirectionalIterator1 first,
BidirectionalIterator1 last,
BidirectionalIterator2 result)
{
while(first != last)
{
*--result = *--last;
} 

return result;
} 


namespace dispatch
{


template<typename DerivedPolicy,
typename RandomAccessIterator1,
typename RandomAccessIterator2>
RandomAccessIterator2 overlapped_copy(hydra_thrust::system::cpp::detail::execution_policy<DerivedPolicy> &,
RandomAccessIterator1 first,
RandomAccessIterator1 last,
RandomAccessIterator2 result)
{
if(first < last && first <= result && result < last)
{
hydra_thrust::detail::sequential_copy_backward(first, last, result + (last - first));
result += (last - first);
} 
else
{
result = hydra_thrust::detail::sequential_copy(first, last, result);
} 

return result;
} 


template<typename DerivedPolicy,
typename RandomAccessIterator1,
typename RandomAccessIterator2>
RandomAccessIterator2 overlapped_copy(hydra_thrust::execution_policy<DerivedPolicy> &exec,
RandomAccessIterator1 first,
RandomAccessIterator1 last,
RandomAccessIterator2 result)
{
typedef typename hydra_thrust::iterator_value<RandomAccessIterator1>::type value_type;

hydra_thrust::detail::temporary_array<value_type, DerivedPolicy> temp(exec, first, last);
return hydra_thrust::copy(exec, temp.begin(), temp.end(), result);
} 

} 


template<typename RandomAccessIterator1,
typename RandomAccessIterator2>
RandomAccessIterator2 overlapped_copy(RandomAccessIterator1 first,
RandomAccessIterator1 last,
RandomAccessIterator2 result)
{
typedef typename hydra_thrust::iterator_system<RandomAccessIterator2>::type System1;
typedef typename hydra_thrust::iterator_system<RandomAccessIterator2>::type System2;

typedef typename hydra_thrust::detail::minimum_system<System1, System2>::type System;

System system;

return hydra_thrust::detail::dispatch::overlapped_copy(system, first, last, result);
} 

} 
} 

