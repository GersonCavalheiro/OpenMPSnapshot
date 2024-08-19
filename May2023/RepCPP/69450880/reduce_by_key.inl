

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/system/tbb/detail/reduce_by_key.h>
#include <hydra/detail/external/hydra_thrust/iterator/reverse_iterator.h>
#include <hydra/detail/external/hydra_thrust/detail/seq.h>
#include <hydra/detail/external/hydra_thrust/system/tbb/detail/execution_policy.h>
#include <hydra/detail/external/hydra_thrust/system/tbb/detail/reduce_intervals.h>
#include <hydra/detail/external/hydra_thrust/detail/minmax.h>
#include <hydra/detail/external/hydra_thrust/detail/temporary_array.h>
#include <hydra/detail/external/hydra_thrust/detail/range/tail_flags.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/tbb_thread.h>
#include <cassert>


namespace hydra_thrust
{
namespace system
{
namespace tbb
{
namespace detail
{
namespace reduce_by_key_detail
{


template<typename L, typename R>
inline L divide_ri(const L x, const R y)
{
return (x + (y - 1)) / y;
}


template<typename InputIterator, typename BinaryFunction, typename OutputIterator = void>
struct partial_sum_type
: hydra_thrust::detail::eval_if<
hydra_thrust::detail::has_result_type<BinaryFunction>::value,
hydra_thrust::detail::result_type<BinaryFunction>,
hydra_thrust::detail::eval_if<
hydra_thrust::detail::is_output_iterator<OutputIterator>::value,
hydra_thrust::iterator_value<InputIterator>,
hydra_thrust::iterator_value<OutputIterator>
>
>
{};


template<typename InputIterator, typename BinaryFunction>
struct partial_sum_type<InputIterator,BinaryFunction,void>
: hydra_thrust::detail::eval_if<
hydra_thrust::detail::has_result_type<BinaryFunction>::value,
hydra_thrust::detail::result_type<BinaryFunction>,
hydra_thrust::iterator_value<InputIterator>
>
{};


template<typename InputIterator1,
typename InputIterator2,
typename BinaryPredicate,
typename BinaryFunction>
hydra_thrust::pair<
InputIterator1,
hydra_thrust::pair<
typename hydra_thrust::iterator_value<InputIterator1>::type,
typename partial_sum_type<InputIterator2,BinaryFunction>::type
>
>
reduce_last_segment_backward(InputIterator1 keys_first,
InputIterator1 keys_last,
InputIterator2 values_first,
BinaryPredicate binary_pred,
BinaryFunction binary_op)
{
typename hydra_thrust::iterator_difference<InputIterator1>::type n = keys_last - keys_first;

hydra_thrust::reverse_iterator<InputIterator1> keys_first_r(keys_last);
hydra_thrust::reverse_iterator<InputIterator1> keys_last_r(keys_first);
hydra_thrust::reverse_iterator<InputIterator2> values_first_r(values_first + n);

typename hydra_thrust::iterator_value<InputIterator1>::type result_key = *keys_first_r;
typename partial_sum_type<InputIterator2,BinaryFunction>::type result_value = *values_first_r;

for(++keys_first_r, ++values_first_r;
(keys_first_r != keys_last_r) && binary_pred(*keys_first_r, result_key);
++keys_first_r, ++values_first_r)
{
result_value = binary_op(result_value, *values_first_r);
}

return hydra_thrust::make_pair(keys_first_r.base(), hydra_thrust::make_pair(result_key, result_value));
}


template<typename InputIterator1,
typename InputIterator2,
typename OutputIterator1,
typename OutputIterator2,
typename BinaryPredicate,
typename BinaryFunction>
hydra_thrust::tuple<
OutputIterator1,
OutputIterator2,
typename hydra_thrust::iterator_value<InputIterator1>::type,
typename partial_sum_type<InputIterator2,BinaryFunction>::type
>
reduce_by_key_with_carry(InputIterator1 keys_first, 
InputIterator1 keys_last,
InputIterator2 values_first,
OutputIterator1 keys_output,
OutputIterator2 values_output,
BinaryPredicate binary_pred,
BinaryFunction binary_op)
{
hydra_thrust::pair<
typename hydra_thrust::iterator_value<InputIterator1>::type,
typename partial_sum_type<InputIterator2,BinaryFunction>::type
> carry;

hydra_thrust::tie(keys_last, carry) = reduce_last_segment_backward(keys_first, keys_last, values_first, binary_pred, binary_op);

hydra_thrust::tie(keys_output, values_output) =
hydra_thrust::reduce_by_key(hydra_thrust::seq, keys_first, keys_last, values_first, keys_output, values_output, binary_pred, binary_op);

return hydra_thrust::make_tuple(keys_output, values_output, carry.first, carry.second);
}


template<typename Iterator>
bool interval_has_carry(size_t interval_idx, size_t interval_size, size_t num_intervals, Iterator tail_flags)
{
return (interval_idx + 1 < num_intervals) ? !tail_flags[(interval_idx + 1) * interval_size - 1] : false;
}


template<typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4, typename Iterator5, typename Iterator6, typename BinaryPredicate, typename BinaryFunction>
struct serial_reduce_by_key_body
{
typedef typename hydra_thrust::iterator_difference<Iterator1>::type size_type;

Iterator1 keys_first;
Iterator2 values_first;
Iterator3 result_offset;
Iterator4 keys_result;
Iterator5 values_result;
Iterator6 carry_result;

size_type n;
size_type interval_size;
size_type num_intervals;

BinaryPredicate binary_pred;
BinaryFunction binary_op;

serial_reduce_by_key_body(Iterator1 keys_first, Iterator2 values_first, Iterator3 result_offset, Iterator4 keys_result, Iterator5 values_result, Iterator6 carry_result, size_type n, size_type interval_size, size_type num_intervals, BinaryPredicate binary_pred, BinaryFunction binary_op)
: keys_first(keys_first), values_first(values_first),
result_offset(result_offset),
keys_result(keys_result),
values_result(values_result),
carry_result(carry_result),
n(n),
interval_size(interval_size),
num_intervals(num_intervals),
binary_pred(binary_pred),
binary_op(binary_op)
{}

void operator()(const ::tbb::blocked_range<size_type> &r) const
{
assert(r.size() == 1);

const size_type interval_idx = r.begin();

const size_type offset_to_first = interval_size * interval_idx;
const size_type offset_to_last = hydra_thrust::min(n, offset_to_first + interval_size);

Iterator1 my_keys_first     = keys_first    + offset_to_first;
Iterator1 my_keys_last      = keys_first    + offset_to_last;
Iterator2 my_values_first   = values_first  + offset_to_first;
Iterator3 my_result_offset  = result_offset + interval_idx;
Iterator4 my_keys_result    = keys_result   + *my_result_offset;
Iterator5 my_values_result  = values_result + *my_result_offset;
Iterator6 my_carry_result   = carry_result  + interval_idx;

typedef typename hydra_thrust::iterator_value<Iterator1>::type key_type;
typedef typename partial_sum_type<Iterator2,BinaryFunction>::type value_type;

hydra_thrust::pair<key_type, value_type> carry;

hydra_thrust::tie(my_keys_result, my_values_result, carry.first, carry.second) =
reduce_by_key_with_carry(my_keys_first,
my_keys_last,
my_values_first,
my_keys_result,
my_values_result,
binary_pred,
binary_op);


hydra_thrust::detail::tail_flags<Iterator1,BinaryPredicate> flags = hydra_thrust::detail::make_tail_flags(keys_first, keys_first + n, binary_pred);

if(interval_has_carry(interval_idx, interval_size, num_intervals, flags.begin()))
{
*my_carry_result = carry.second;
}
else
{
*my_keys_result = carry.first;
*my_values_result = carry.second;
}
}
};


template<typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4, typename Iterator5, typename Iterator6, typename BinaryPredicate, typename BinaryFunction>
serial_reduce_by_key_body<Iterator1,Iterator2,Iterator3,Iterator4,Iterator5,Iterator6,BinaryPredicate,BinaryFunction>
make_serial_reduce_by_key_body(Iterator1 keys_first, Iterator2 values_first, Iterator3 result_offset, Iterator4 keys_result, Iterator5 values_result, Iterator6 carry_result, typename hydra_thrust::iterator_difference<Iterator1>::type n, size_t interval_size, size_t num_intervals, BinaryPredicate binary_pred, BinaryFunction binary_op)
{
return serial_reduce_by_key_body<Iterator1,Iterator2,Iterator3,Iterator4,Iterator5,Iterator6,BinaryPredicate,BinaryFunction>(keys_first, values_first, result_offset, keys_result, values_result, carry_result, n, interval_size, num_intervals, binary_pred, binary_op);
}


} 


template<typename DerivedPolicy, typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4, typename BinaryPredicate, typename BinaryFunction>
hydra_thrust::pair<Iterator3,Iterator4>
reduce_by_key(hydra_thrust::tbb::execution_policy<DerivedPolicy> &exec,
Iterator1 keys_first, Iterator1 keys_last, 
Iterator2 values_first,
Iterator3 keys_result,
Iterator4 values_result,
BinaryPredicate binary_pred,
BinaryFunction binary_op)
{

typedef typename hydra_thrust::iterator_difference<Iterator1>::type difference_type;
difference_type n = keys_last - keys_first;
if(n == 0) return hydra_thrust::make_pair(keys_result, values_result);

const difference_type parallelism_threshold = 10000;

if(n < parallelism_threshold)
{
return hydra_thrust::reduce_by_key(hydra_thrust::seq, keys_first, keys_last, values_first, keys_result, values_result, binary_pred, binary_op);
}

const unsigned int p = hydra_thrust::max<unsigned int>(1u, ::tbb::tbb_thread::hardware_concurrency());

const unsigned int subscription_rate = 1;
difference_type interval_size = hydra_thrust::min<difference_type>(parallelism_threshold, hydra_thrust::max<difference_type>(n, n / (subscription_rate * p)));
difference_type num_intervals = reduce_by_key_detail::divide_ri(n, interval_size);

hydra_thrust::detail::temporary_array<difference_type, DerivedPolicy> interval_output_offsets(0, exec, num_intervals + 1);

hydra_thrust::detail::tail_flags<Iterator1,BinaryPredicate> tail_flags = hydra_thrust::detail::make_tail_flags(keys_first, keys_last, binary_pred);
hydra_thrust::system::tbb::detail::reduce_intervals(exec, tail_flags.begin(), tail_flags.end(), interval_size, interval_output_offsets.begin() + 1, hydra_thrust::plus<size_t>());
interval_output_offsets[0] = 0;

hydra_thrust::inclusive_scan(hydra_thrust::seq,
interval_output_offsets.begin() + 1, interval_output_offsets.end(), 
interval_output_offsets.begin() + 1);

typedef typename reduce_by_key_detail::partial_sum_type<Iterator2,BinaryFunction>::type carry_type;
hydra_thrust::detail::temporary_array<carry_type, DerivedPolicy> carries(0, exec, num_intervals - 1);

::tbb::parallel_for(::tbb::blocked_range<difference_type>(0, num_intervals, 1),
reduce_by_key_detail::make_serial_reduce_by_key_body(keys_first, values_first, interval_output_offsets.begin(), keys_result, values_result, carries.begin(), n, interval_size, num_intervals, binary_pred, binary_op),
::tbb::simple_partitioner());

difference_type size_of_result = interval_output_offsets[num_intervals];

for(typename hydra_thrust::detail::temporary_array<carry_type,DerivedPolicy>::size_type i = 0; i < carries.size(); ++i)
{
if(reduce_by_key_detail::interval_has_carry(i, interval_size, num_intervals, tail_flags.begin()))
{
difference_type output_idx = interval_output_offsets[i+1];

values_result[output_idx] = binary_op(values_result[output_idx], carries[i]);
}
}

return hydra_thrust::make_pair(keys_result + size_of_result, values_result + size_of_result);
}


} 
} 
} 
} 

