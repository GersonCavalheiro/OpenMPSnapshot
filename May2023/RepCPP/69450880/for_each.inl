




#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/static_assert.h>
#include <hydra/detail/external/hydra_thrust/distance.h>
#include <hydra/detail/external/hydra_thrust/detail/function.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/distance.h>
#include <hydra/detail/external/hydra_thrust/for_each.h>

namespace hydra_thrust
{
namespace system
{
namespace omp
{
namespace detail
{

template<typename DerivedPolicy,
typename RandomAccessIterator,
typename Size,
typename UnaryFunction>
RandomAccessIterator for_each_n(execution_policy<DerivedPolicy> &,
RandomAccessIterator first,
Size n,
UnaryFunction f)
{
HYDRA_THRUST_STATIC_ASSERT_MSG(
(hydra_thrust::detail::depend_on_instantiation<
RandomAccessIterator, (HYDRA_THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == HYDRA_THRUST_TRUE)
>::value)
, "OpenMP compiler support is not enabled"
);

if (n <= 0) return first;  

hydra_thrust::detail::wrapped_function<UnaryFunction,void> wrapped_f(f);

#if (HYDRA_THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == HYDRA_THRUST_TRUE)
typedef typename hydra_thrust::iterator_difference<RandomAccessIterator>::type DifferenceType;
DifferenceType signed_n = n;
#pragma omp parallel for
for(DifferenceType i = 0;
i < signed_n;
++i)
{
RandomAccessIterator temp = first + i;
wrapped_f(*temp);
}
#endif 

return first + n;
} 

template<typename DerivedPolicy,
typename RandomAccessIterator,
typename UnaryFunction>
RandomAccessIterator for_each(execution_policy<DerivedPolicy> &s,
RandomAccessIterator first,
RandomAccessIterator last,
UnaryFunction f)
{
return omp::detail::for_each_n(s, first, hydra_thrust::distance(first,last), f);
} 

} 
} 
} 
} 

