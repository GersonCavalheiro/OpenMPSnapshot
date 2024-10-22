#pragma once

#include <cstddef> 
#include <type_traits> 

#include <nlohmann/detail/boolean_operators.hpp>

namespace nlohmann
{
namespace detail
{
template<bool B, typename T = void>
using enable_if_t = typename std::enable_if<B, T>::type;

template<typename T>
using uncvref_t = typename std::remove_cv<typename std::remove_reference<T>::type>::type;

template<std::size_t... Ints>
struct index_sequence
{
using type = index_sequence;
using value_type = std::size_t;
static constexpr std::size_t size() noexcept
{
return sizeof...(Ints);
}
};

template<class Sequence1, class Sequence2>
struct merge_and_renumber;

template<std::size_t... I1, std::size_t... I2>
struct merge_and_renumber<index_sequence<I1...>, index_sequence<I2...>>
: index_sequence < I1..., (sizeof...(I1) + I2)... > {};

template<std::size_t N>
struct make_index_sequence
: merge_and_renumber < typename make_index_sequence < N / 2 >::type,
typename make_index_sequence < N - N / 2 >::type > {};

template<> struct make_index_sequence<0> : index_sequence<> {};
template<> struct make_index_sequence<1> : index_sequence<0> {};

template<typename... Ts>
using index_sequence_for = make_index_sequence<sizeof...(Ts)>;

template<unsigned N> struct priority_tag : priority_tag < N - 1 > {};
template<> struct priority_tag<0> {};

template<typename T>
struct static_const
{
static constexpr T value{};
};

template<typename T>
constexpr T static_const<T>::value;
}  
}  
