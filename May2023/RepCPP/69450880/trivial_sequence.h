




#pragma once

#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/execution_policy.h>
#include <hydra/detail/external/hydra_thrust/detail/temporary_array.h>
#include <hydra/detail/external/hydra_thrust/type_traits/is_contiguous_iterator.h>

namespace hydra_thrust
{

namespace detail
{

template<typename Iterator, typename DerivedPolicy, typename is_trivial> struct _trivial_sequence { };

template<typename Iterator, typename DerivedPolicy>
struct _trivial_sequence<Iterator, DerivedPolicy, hydra_thrust::detail::true_type>
{
typedef Iterator iterator_type;
Iterator first, last;

__host__ __device__
_trivial_sequence(hydra_thrust::execution_policy<DerivedPolicy> &, Iterator _first, Iterator _last) : first(_first), last(_last)
{
}

__host__ __device__
iterator_type begin() { return first; }

__host__ __device__
iterator_type end()   { return last; }
};

template<typename Iterator, typename DerivedPolicy>
struct _trivial_sequence<Iterator, DerivedPolicy, hydra_thrust::detail::false_type>
{
typedef typename hydra_thrust::iterator_value<Iterator>::type iterator_value;
typedef typename hydra_thrust::detail::temporary_array<iterator_value, DerivedPolicy>::iterator iterator_type;

hydra_thrust::detail::temporary_array<iterator_value, DerivedPolicy> buffer;

__host__ __device__
_trivial_sequence(hydra_thrust::execution_policy<DerivedPolicy> &exec, Iterator first, Iterator last)
: buffer(exec, first, last)
{
}

__host__ __device__
iterator_type begin() { return buffer.begin(); }

__host__ __device__
iterator_type end()   { return buffer.end(); }
};

template <typename Iterator, typename DerivedPolicy>
struct trivial_sequence
: detail::_trivial_sequence<Iterator, DerivedPolicy, typename hydra_thrust::is_contiguous_iterator<Iterator>::type>
{
typedef _trivial_sequence<Iterator, DerivedPolicy, typename hydra_thrust::is_contiguous_iterator<Iterator>::type> super_t;

__host__ __device__
trivial_sequence(hydra_thrust::execution_policy<DerivedPolicy> &exec, Iterator first, Iterator last) : super_t(exec, first, last) { }
};

} 

} 

