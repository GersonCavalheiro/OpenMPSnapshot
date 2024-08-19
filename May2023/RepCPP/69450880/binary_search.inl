

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/pair.h>
#include <hydra/detail/external/hydra_thrust/detail/function.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>

namespace hydra_thrust
{

namespace system
{

namespace detail
{

namespace generic
{

namespace scalar
{

template<typename RandomAccessIterator, typename Size, typename T, typename BinaryPredicate>
__host__ __device__
RandomAccessIterator lower_bound_n(RandomAccessIterator first,
Size n,
const T &val,
BinaryPredicate comp)
{
hydra_thrust::detail::wrapped_function<
BinaryPredicate,
bool
> wrapped_comp(comp);

Size start = 0, i;
while(start < n)
{
i = (start + n) / 2;
if(wrapped_comp(first[i], val))
{
start = i + 1;
}
else
{
n = i;
}
} 

return first + start;
}


template<typename RandomAccessIterator, typename T, typename BinaryPredicate>
__host__ __device__
RandomAccessIterator lower_bound(RandomAccessIterator first, RandomAccessIterator last,
const T &val,
BinaryPredicate comp)
{
typename hydra_thrust::iterator_difference<RandomAccessIterator>::type n = last - first;
return lower_bound_n(first, n, val, comp);
}

template<typename RandomAccessIterator, typename Size, typename T, typename BinaryPredicate>
__host__ __device__
RandomAccessIterator upper_bound_n(RandomAccessIterator first,
Size n,
const T &val,
BinaryPredicate comp)
{
hydra_thrust::detail::wrapped_function<
BinaryPredicate,
bool
> wrapped_comp(comp);

Size start = 0, i;
while(start < n)
{
i = (start + n) / 2;
if(wrapped_comp(val, first[i]))
{
n = i;
}
else
{
start = i + 1;
}
} 

return first + start;
}

template<typename RandomAccessIterator, typename T, typename BinaryPredicate>
__host__ __device__
RandomAccessIterator upper_bound(RandomAccessIterator first, RandomAccessIterator last,
const T &val,
BinaryPredicate comp)
{
typename hydra_thrust::iterator_difference<RandomAccessIterator>::type n = last - first;
return upper_bound_n(first, n, val, comp);
}

template<typename RandomAccessIterator, typename T, typename BinaryPredicate>
__host__ __device__
pair<RandomAccessIterator,RandomAccessIterator>
equal_range(RandomAccessIterator first, RandomAccessIterator last,
const T &val,
BinaryPredicate comp)
{
RandomAccessIterator lb = hydra_thrust::system::detail::generic::scalar::lower_bound(first, last, val, comp);
return hydra_thrust::make_pair(lb, hydra_thrust::system::detail::generic::scalar::upper_bound(lb, last, val, comp));
}


template<typename RandomAccessIterator, typename T, typename Compare>
__host__ __device__
bool binary_search(RandomAccessIterator first, RandomAccessIterator last, const T &value, Compare comp)
{
RandomAccessIterator iter = hydra_thrust::system::detail::generic::scalar::lower_bound(first, last, value, comp);

hydra_thrust::detail::wrapped_function<
Compare,
bool
> wrapped_comp(comp);

return iter != last && !wrapped_comp(value,*iter);
}

} 

} 

} 

} 

} 

#include <hydra/detail/external/hydra_thrust/system/detail/generic/scalar/binary_search.inl>

