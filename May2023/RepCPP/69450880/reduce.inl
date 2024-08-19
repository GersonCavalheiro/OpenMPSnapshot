


#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/function.h>
#include <hydra/detail/external/hydra_thrust/detail/static_assert.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/distance.h>
#include <hydra/detail/external/hydra_thrust/reduce.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

namespace hydra_thrust
{
namespace system
{
namespace tbb
{
namespace detail
{
namespace reduce_detail
{

template<typename RandomAccessIterator,
typename OutputType,
typename BinaryFunction>
struct body
{
RandomAccessIterator first;
OutputType sum;
bool first_call;  
hydra_thrust::detail::wrapped_function<BinaryFunction,OutputType> binary_op;

body(RandomAccessIterator first, OutputType init, BinaryFunction binary_op)
: first(first), sum(init), first_call(true), binary_op(binary_op)
{}

body(body& b, ::tbb::split)
: first(b.first), sum(b.sum), first_call(true), binary_op(b.binary_op)
{}

template <typename Size>
void operator()(const ::tbb::blocked_range<Size> &r)
{

if (r.empty()) return; 

RandomAccessIterator iter = first + r.begin();

OutputType temp = hydra_thrust::raw_reference_cast(*iter);

++iter;

for (Size i = r.begin() + 1; i != r.end(); ++i, ++iter)
temp = binary_op(temp, *iter);


if (first_call)
{
first_call = false;
sum = temp;
}
else
{
sum = binary_op(sum, temp);
}
} 

void join(body& b)
{
sum = binary_op(sum, b.sum);
}
}; 

} 


template<typename DerivedPolicy,
typename InputIterator, 
typename OutputType,
typename BinaryFunction>
OutputType reduce(execution_policy<DerivedPolicy> &exec,
InputIterator begin,
InputIterator end,
OutputType init,
BinaryFunction binary_op)
{
typedef typename hydra_thrust::iterator_difference<InputIterator>::type Size; 

Size n = hydra_thrust::distance(begin, end);

if (n == 0)
{
return init;
}
else
{
typedef typename reduce_detail::body<InputIterator,OutputType,BinaryFunction> Body;
Body reduce_body(begin, init, binary_op);
::tbb::parallel_reduce(::tbb::blocked_range<Size>(0,n), reduce_body);
return binary_op(init, reduce_body.sum);
}
}


} 
} 
} 
} 

