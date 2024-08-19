


#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/cpp11_required.h>

#if HYDRA_THRUST_CPP_DIALECT >= 2011

#include <type_traits>
#include <utility>
#include <cstdint>
#include <utility>

HYDRA_THRUST_BEGIN_NS

#if HYDRA_THRUST_CPP_DIALECT >= 2014

template <typename T, T... Is>
using integer_sequence = std::integer_sequence<T, Is...>;

template <std::size_t... Is>
using index_sequence = std::index_sequence<Is...>;

template <typename T, std::size_t N>
using make_integer_sequence = std::make_integer_sequence<T, N>;

template <std::size_t N>
using make_index_sequence = std::make_index_sequence<N>;


#else 

template <typename T, T... Is>
struct integer_sequence;

template <std::size_t... Is>
using index_sequence = integer_sequence<std::size_t, Is...>;


namespace detail
{

template <typename Sequence0, typename Sequence1>
struct merge_and_renumber_integer_sequences_impl;
template <typename Sequence0, typename Sequence1>
using merge_and_renumber_integer_sequences =
typename merge_and_renumber_integer_sequences_impl<
Sequence0, Sequence1
>::type;

template <typename T, std::size_t N>
struct make_integer_sequence_impl;


} 


template <typename T, std::size_t N>
using make_integer_sequence =
typename detail::make_integer_sequence_impl<T, N>::type;

template <std::size_t N>
using make_index_sequence =
make_integer_sequence<std::size_t, N>;


template <typename T, T... Is>
struct integer_sequence
{
using type = integer_sequence;
using value_type = T;
using size_type = std::size_t;

__host__ __device__
static constexpr size_type size() noexcept
{
return sizeof...(Is);
}
};

namespace detail
{

template <typename T, T... Is0, T... Is1>
struct merge_and_renumber_integer_sequences_impl<
integer_sequence<T, Is0...>, integer_sequence<T, Is1...>
>
{
using type = integer_sequence<T, Is0..., (sizeof...(Is0) + Is1)...>;
};


template <typename T, std::size_t N>
struct make_integer_sequence_impl
{
using type = merge_and_renumber_integer_sequences<
make_integer_sequence<T, N / 2>
, make_integer_sequence<T, N - N / 2>
>;
};

template <typename T>
struct make_integer_sequence_impl<T, 0>
{
using type = integer_sequence<T>;
};

template <typename T>
struct make_integer_sequence_impl<T, 1>
{
using type = integer_sequence<T, 0>;
};

} 

#endif 


namespace detail
{

template <typename Sequence0, typename Sequence1>
struct merge_and_renumber_reversed_integer_sequences_impl;
template <typename Sequence0, typename Sequence1>
using merge_and_renumber_reversed_integer_sequences =
typename merge_and_renumber_reversed_integer_sequences_impl<
Sequence0, Sequence1
>::type;

template <typename T, std::size_t N>
struct make_reversed_integer_sequence_impl;

template <typename T, T I, typename Sequence> 
struct integer_sequence_push_front_impl;

template <typename T, T I, typename Sequence> 
struct integer_sequence_push_back_impl;

}


template <typename T, std::size_t N>
using make_reversed_integer_sequence =
typename detail::make_reversed_integer_sequence_impl<T, N>::type;

template <std::size_t N>
using make_reversed_index_sequence =
make_reversed_integer_sequence<std::size_t, N>;

template <typename T, T I, typename Sequence> 
using integer_sequence_push_front =
typename detail::integer_sequence_push_front_impl<T, I, Sequence>::type;

template <typename T, T I, typename Sequence> 
using integer_sequence_push_back =
typename detail::integer_sequence_push_back_impl<T, I, Sequence>::type;


namespace detail
{

template <typename T, T... Is0, T... Is1>
struct merge_and_renumber_reversed_integer_sequences_impl<
integer_sequence<T, Is0...>, integer_sequence<T, Is1...>
>
{
using type = integer_sequence<T, (sizeof...(Is1) + Is0)..., Is1...>;
};


template <typename T, std::size_t N>
struct make_reversed_integer_sequence_impl
{
using type = merge_and_renumber_reversed_integer_sequences<
make_reversed_integer_sequence<T, N / 2>
, make_reversed_integer_sequence<T, N - N / 2>
>;
};


template <typename T>
struct make_reversed_integer_sequence_impl<T, 0>
{
using type = integer_sequence<T>;
};

template <typename T>
struct make_reversed_integer_sequence_impl<T, 1>
{
using type = integer_sequence<T, 0>;
};


template <typename T, T I0, T... Is> 
struct integer_sequence_push_front_impl<T, I0, integer_sequence<T, Is...> >
{
using type = integer_sequence<T, I0, Is...>;
};


template <typename T, T I0, T... Is> 
struct integer_sequence_push_back_impl<T, I0, integer_sequence<T, Is...> >
{
using type = integer_sequence<T, Is..., I0>;
};


} 

HYDRA_THRUST_END_NS

#endif 

