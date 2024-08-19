


#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/system/omp/detail/reduce_intervals.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/function.h>
#include <hydra/detail/external/hydra_thrust/detail/cstdint.h>

namespace hydra_thrust
{
namespace system
{
namespace omp
{
namespace detail
{

template <typename DerivedPolicy,
typename InputIterator,
typename OutputIterator,
typename BinaryFunction,
typename Decomposition>
void reduce_intervals(execution_policy<DerivedPolicy> &,
InputIterator input,
OutputIterator output,
BinaryFunction binary_op,
Decomposition decomp)
{
HYDRA_THRUST_STATIC_ASSERT_MSG(
(hydra_thrust::detail::depend_on_instantiation<
InputIterator, (HYDRA_THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == HYDRA_THRUST_TRUE)
>::value)
, "OpenMP compiler support is not enabled"
);

#if (HYDRA_THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == HYDRA_THRUST_TRUE)
typedef typename hydra_thrust::iterator_value<OutputIterator>::type OutputType;

hydra_thrust::detail::wrapped_function<BinaryFunction,OutputType> wrapped_binary_op(binary_op);

typedef hydra_thrust::detail::intptr_t index_type;

index_type n = static_cast<index_type>(decomp.size());

#if (HYDRA_THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == HYDRA_THRUST_TRUE)
# pragma omp parallel for
#endif 
for(index_type i = 0; i < n; i++)
{
InputIterator begin = input + decomp[i].begin();
InputIterator end   = input + decomp[i].end();

if (begin != end)
{
OutputType sum = hydra_thrust::raw_reference_cast(*begin);

++begin;

while (begin != end)
{
sum = wrapped_binary_op(sum, *begin);
++begin;
}

OutputIterator tmp = output + i;
*tmp = sum;
}
}
#endif 
}

} 
} 
} 
} 

