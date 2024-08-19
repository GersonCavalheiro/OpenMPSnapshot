

#pragma once

#include <hydra/detail/external/hydra_thrust/tuple.h>

namespace hydra_thrust
{

namespace detail
{

template<typename Tuple,
template<typename> class UnaryMetaFunction,
typename IndexSequence = hydra_thrust::__make_index_sequence<hydra_thrust::tuple_size<Tuple>::value>>
struct tuple_meta_transform;

template<typename Tuple,
template<typename> class UnaryMetaFunction,
size_t... I>
struct tuple_meta_transform<Tuple, UnaryMetaFunction, hydra_thrust::__index_sequence<I...>>
{
typedef hydra_thrust::tuple<
typename UnaryMetaFunction<typename hydra_thrust::tuple_element<I,Tuple>::type>::type...
> type;
};

} 

} 

