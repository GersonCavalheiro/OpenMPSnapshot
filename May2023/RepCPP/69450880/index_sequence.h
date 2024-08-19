



#pragma once

#include <cstddef> 

namespace hydra_thrust
{

template<size_t... I> struct __index_sequence {};

template<size_t Start, typename Indices, size_t End>
struct __make_index_sequence_impl;

template<size_t Start, size_t... Indices, size_t End>
struct __make_index_sequence_impl<
Start,
__index_sequence<Indices...>,
End
>
{
typedef typename __make_index_sequence_impl<
Start + 1,
__index_sequence<Indices..., Start>,
End
>::type type;
};

template<size_t End, size_t... Indices>
struct __make_index_sequence_impl<End, __index_sequence<Indices...>, End>
{
typedef __index_sequence<Indices...> type;
};

template<size_t N>
using __make_index_sequence = typename __make_index_sequence_impl<0, __index_sequence<>, N>::type;

template<class... T>
using __index_sequence_for = __make_index_sequence<sizeof...(T)>;

} 
