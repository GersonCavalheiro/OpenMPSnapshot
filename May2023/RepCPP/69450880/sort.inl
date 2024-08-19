


#include <hydra/detail/external/hydra_thrust/detail/config.h>

#if (HYDRA_THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == HYDRA_THRUST_TRUE)
#include <omp.h>
#endif 

#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/system/omp/detail/default_decomposition.h>
#include <hydra/detail/external/hydra_thrust/system/detail/generic/select_system.h>
#include <hydra/detail/external/hydra_thrust/sort.h>
#include <hydra/detail/external/hydra_thrust/merge.h>
#include <hydra/detail/external/hydra_thrust/detail/seq.h>
#include <hydra/detail/external/hydra_thrust/detail/temporary_array.h>

namespace hydra_thrust
{
namespace system
{
namespace omp
{
namespace detail
{
namespace sort_detail
{


template<typename DerivedPolicy,
typename RandomAccessIterator,
typename StrictWeakOrdering>
void inplace_merge(execution_policy<DerivedPolicy> &exec,
RandomAccessIterator first,
RandomAccessIterator middle,
RandomAccessIterator last,
StrictWeakOrdering comp)
{
typedef typename hydra_thrust::iterator_value<RandomAccessIterator>::type value_type;

hydra_thrust::detail::temporary_array<value_type,DerivedPolicy> a(exec, first, middle);
hydra_thrust::detail::temporary_array<value_type,DerivedPolicy> b(exec, middle, last);

hydra_thrust::merge(hydra_thrust::seq, a.begin(), a.end(), b.begin(), b.end(), first, comp);
}


template<typename DerivedPolicy,
typename RandomAccessIterator1,
typename RandomAccessIterator2,
typename StrictWeakOrdering>
void inplace_merge_by_key(execution_policy<DerivedPolicy> &exec,
RandomAccessIterator1 first1,
RandomAccessIterator1 middle1,
RandomAccessIterator1 last1,
RandomAccessIterator2 first2,
StrictWeakOrdering comp)
{
typedef typename hydra_thrust::iterator_value<RandomAccessIterator1>::type value_type1;
typedef typename hydra_thrust::iterator_value<RandomAccessIterator2>::type value_type2;

RandomAccessIterator2 middle2 = first2 + (middle1 - first1);
RandomAccessIterator2 last2   = first2 + (last1   - first1);

hydra_thrust::detail::temporary_array<value_type1,DerivedPolicy> lhs1(exec, first1, middle1);
hydra_thrust::detail::temporary_array<value_type1,DerivedPolicy> rhs1(exec, middle1, last1);
hydra_thrust::detail::temporary_array<value_type2,DerivedPolicy> lhs2(exec, first2, middle2);
hydra_thrust::detail::temporary_array<value_type2,DerivedPolicy> rhs2(exec, middle2, last2);

hydra_thrust::merge_by_key(hydra_thrust::seq,
lhs1.begin(), lhs1.end(),
rhs1.begin(), rhs1.end(),
lhs2.begin(), rhs2.begin(),
first1, first2,
comp);
}


} 


template<typename DerivedPolicy,
typename RandomAccessIterator,
typename StrictWeakOrdering>
void stable_sort(execution_policy<DerivedPolicy> &exec,
RandomAccessIterator first,
RandomAccessIterator last,
StrictWeakOrdering comp)
{
HYDRA_THRUST_STATIC_ASSERT_MSG(
(hydra_thrust::detail::depend_on_instantiation<
RandomAccessIterator, (HYDRA_THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == HYDRA_THRUST_TRUE)
>::value)
, "OpenMP compiler support is not enabled"
);

#if (HYDRA_THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == HYDRA_THRUST_TRUE)
typedef typename hydra_thrust::iterator_difference<RandomAccessIterator>::type IndexType;

if(first == last)
return;

#pragma omp parallel
{
hydra_thrust::system::detail::internal::uniform_decomposition<IndexType> decomp(last - first, 1, omp_get_num_threads());

IndexType p_i = omp_get_thread_num();

if(p_i < decomp.size())
{
hydra_thrust::stable_sort(hydra_thrust::seq,
first + decomp[p_i].begin(),
first + decomp[p_i].end(),
comp);
}

#pragma omp barrier

;

IndexType nseg = decomp.size();
IndexType h = 2;

IndexType a=p_i, b=p_i, c=p_i+1;

while(nseg>1)
{
if(c >= decomp.size())
c = decomp.size() - 1;

if((p_i % h) == 0 && c > b)
{
sort_detail::inplace_merge(exec,
first + decomp[a].begin(),
first + decomp[b].end(),
first + decomp[c].end(),
comp);

b = c;
c += h;
}

nseg = (nseg + 1) / 2;
h *= 2;

#pragma omp barrier
}
}
#endif 
}


template<typename DerivedPolicy,
typename RandomAccessIterator1,
typename RandomAccessIterator2,
typename StrictWeakOrdering>
void stable_sort_by_key(execution_policy<DerivedPolicy> &exec,
RandomAccessIterator1 keys_first,
RandomAccessIterator1 keys_last,
RandomAccessIterator2 values_first,
StrictWeakOrdering comp)
{
HYDRA_THRUST_STATIC_ASSERT_MSG(
(hydra_thrust::detail::depend_on_instantiation<
RandomAccessIterator1, (HYDRA_THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == HYDRA_THRUST_TRUE)
>::value)
, "OpenMP compiler support is not enabled"
);

#if (HYDRA_THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == HYDRA_THRUST_TRUE)
typedef typename hydra_thrust::iterator_difference<RandomAccessIterator1>::type IndexType;

if(keys_first == keys_last)
return;

#pragma omp parallel
{
hydra_thrust::system::detail::internal::uniform_decomposition<IndexType> decomp(keys_last - keys_first, 1, omp_get_num_threads());

IndexType p_i = omp_get_thread_num();

if(p_i < decomp.size())
{
hydra_thrust::stable_sort_by_key(hydra_thrust::seq,
keys_first + decomp[p_i].begin(),
keys_first + decomp[p_i].end(),
values_first + decomp[p_i].begin(),
comp);
}

#pragma omp barrier

;

IndexType nseg = decomp.size();
IndexType h = 2;

IndexType a=p_i, b=p_i, c=p_i+1;

while(nseg>1)
{
if(c >= decomp.size())
c = decomp.size() - 1;

if((p_i % h) == 0 && c > b)
{
sort_detail::inplace_merge_by_key(exec,
keys_first + decomp[a].begin(),
keys_first + decomp[b].end(),
keys_first + decomp[c].end(),
values_first + decomp[a].begin(),
comp);

b = c;
c += h;
}

nseg = (nseg + 1) / 2;
h *= 2;

#pragma omp barrier
}
}
#endif 
}


} 
} 
} 
} 

