


#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/system/detail/generic/transform_scan.h>
#include <hydra/detail/external/hydra_thrust/scan.h>
#include <hydra/detail/external/hydra_thrust/iterator/transform_iterator.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits/function_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits/iterator/is_output_iterator.h>

namespace hydra_thrust
{
namespace system
{
namespace detail
{
namespace generic
{


template<typename ExecutionPolicy,
typename InputIterator,
typename OutputIterator,
typename UnaryFunction,
typename BinaryFunction>
__host__ __device__
OutputIterator transform_inclusive_scan(hydra_thrust::execution_policy<ExecutionPolicy> &exec,
InputIterator first,
InputIterator last,
OutputIterator result,
UnaryFunction unary_op,
BinaryFunction binary_op)
{

typedef typename hydra_thrust::detail::eval_if<
hydra_thrust::detail::has_result_type<UnaryFunction>::value,
hydra_thrust::detail::result_type<UnaryFunction>,
hydra_thrust::detail::eval_if<
hydra_thrust::detail::is_output_iterator<OutputIterator>::value,
hydra_thrust::iterator_value<InputIterator>,
hydra_thrust::iterator_value<OutputIterator>
>
>::type ValueType;

hydra_thrust::transform_iterator<UnaryFunction, InputIterator, ValueType> _first(first, unary_op);
hydra_thrust::transform_iterator<UnaryFunction, InputIterator, ValueType> _last(last, unary_op);

return hydra_thrust::inclusive_scan(exec, _first, _last, result, binary_op);
} 


template<typename ExecutionPolicy,
typename InputIterator,
typename OutputIterator,
typename UnaryFunction,
typename T,
typename AssociativeOperator>
__host__ __device__
OutputIterator transform_exclusive_scan(hydra_thrust::execution_policy<ExecutionPolicy> &exec,
InputIterator first,
InputIterator last,
OutputIterator result,
UnaryFunction unary_op,
T init,
AssociativeOperator binary_op)
{

typedef typename hydra_thrust::detail::eval_if<
hydra_thrust::detail::has_result_type<UnaryFunction>::value,
hydra_thrust::detail::result_type<UnaryFunction>,
hydra_thrust::detail::eval_if<
hydra_thrust::detail::is_output_iterator<OutputIterator>::value,
hydra_thrust::iterator_value<InputIterator>,
hydra_thrust::iterator_value<OutputIterator>
>
>::type ValueType;

hydra_thrust::transform_iterator<UnaryFunction, InputIterator, ValueType> _first(first, unary_op);
hydra_thrust::transform_iterator<UnaryFunction, InputIterator, ValueType> _last(last, unary_op);

return hydra_thrust::exclusive_scan(exec, _first, _last, result, init, binary_op);
} 


} 
} 
} 
} 

