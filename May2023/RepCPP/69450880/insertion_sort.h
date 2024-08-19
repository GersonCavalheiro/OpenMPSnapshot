

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>

#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/function.h>
#include <hydra/detail/external/hydra_thrust/system/detail/sequential/copy_backward.h>

namespace hydra_thrust
{
namespace system
{
namespace detail
{
namespace sequential
{


__hydra_thrust_exec_check_disable__
template<typename RandomAccessIterator,
typename StrictWeakOrdering>
__host__ __device__
void insertion_sort(RandomAccessIterator first,
RandomAccessIterator last,
StrictWeakOrdering comp)
{
typedef typename hydra_thrust::iterator_value<RandomAccessIterator>::type value_type;

if(first == last) return;

hydra_thrust::detail::wrapped_function<
StrictWeakOrdering,
bool
> wrapped_comp(comp);

for(RandomAccessIterator i = first + 1; i != last; ++i)
{
value_type tmp = *i;

if(wrapped_comp(tmp, *first))
{
sequential::copy_backward(first, i, i + 1);

*first = tmp;
}
else
{
RandomAccessIterator j = i;
RandomAccessIterator k = i - 1;

while(wrapped_comp(tmp, *k))
{
*j = *k;
j = k;
--k;
}

*j = tmp;
}
}
}


__hydra_thrust_exec_check_disable__
template<typename RandomAccessIterator1,
typename RandomAccessIterator2,
typename StrictWeakOrdering>
__host__ __device__
void insertion_sort_by_key(RandomAccessIterator1 first1,
RandomAccessIterator1 last1,
RandomAccessIterator2 first2,
StrictWeakOrdering comp)
{
typedef typename hydra_thrust::iterator_value<RandomAccessIterator1>::type value_type1;
typedef typename hydra_thrust::iterator_value<RandomAccessIterator2>::type value_type2;

if(first1 == last1) return;

hydra_thrust::detail::wrapped_function<
StrictWeakOrdering,
bool
> wrapped_comp(comp);

RandomAccessIterator1 i1 = first1 + 1;
RandomAccessIterator2 i2 = first2 + 1;

for(; i1 != last1; ++i1, ++i2)
{
value_type1 tmp1 = *i1;
value_type2 tmp2 = *i2;

if(wrapped_comp(tmp1, *first1))
{
sequential::copy_backward(first1, i1, i1 + 1);
sequential::copy_backward(first2, i2, i2 + 1);

*first1 = tmp1;
*first2 = tmp2;
}
else
{
RandomAccessIterator1 j1 = i1;
RandomAccessIterator1 k1 = i1 - 1;

RandomAccessIterator2 j2 = i2;
RandomAccessIterator2 k2 = i2 - 1;

while(wrapped_comp(tmp1, *k1))
{
*j1 = *k1;
*j2 = *k2;

j1 = k1;
j2 = k2;

--k1;
--k2;
}

*j1 = tmp1;
*j2 = tmp2;
}
}
}


} 
} 
} 
} 

