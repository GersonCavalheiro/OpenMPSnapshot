

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/cpp11_required.h>

#if HYDRA_THRUST_CPP_DIALECT >= 2011

#include <hydra/detail/external/hydra_thrust/detail/type_deduction.h>
#include <hydra/detail/external/hydra_thrust/type_traits/integer_sequence.h>

#include <tuple>

HYDRA_THRUST_BEGIN_NS

template <typename Tuple, std::size_t... Is>
auto tuple_subset(Tuple&& t, index_sequence<Is...>)
HYDRA_THRUST_DECLTYPE_RETURNS(std::make_tuple(std::get<Is>(HYDRA_THRUST_FWD(t))...));

namespace detail
{

template <typename Tuple, typename F, std::size_t... Is>
void tuple_for_each_impl(Tuple&& t, F&& f, index_sequence<Is...>)
{
auto l = { (f(std::get<Is>(t)), 0)... };
HYDRA_THRUST_UNUSED_VAR(l);
}

template <typename Tuple, typename F, std::size_t... Is>
auto tuple_transform_impl(Tuple&& t, F&& f, index_sequence<Is...>)
HYDRA_THRUST_DECLTYPE_RETURNS(std::make_tuple(f(std::get<Is>(t))...));

} 

template <typename... Ts, typename F>
auto tuple_for_each(std::tuple<Ts...>& t, F&& f)
HYDRA_THRUST_DECLTYPE_RETURNS(
detail::tuple_for_each_impl(
t
, HYDRA_THRUST_FWD(f)
, make_index_sequence<sizeof...(Ts)>{}
)
);
template <typename... Ts, typename F>
auto tuple_for_each(std::tuple<Ts...> const& t, F&& f)
HYDRA_THRUST_DECLTYPE_RETURNS(
detail::tuple_for_each_impl(
t
, HYDRA_THRUST_FWD(f)
, make_index_sequence<sizeof...(Ts)>{}
)
);
template <typename... Ts, typename F>
auto tuple_for_each(std::tuple<Ts...>&& t, F&& f)
HYDRA_THRUST_DECLTYPE_RETURNS(
detail::tuple_for_each_impl(
std::move(t)
, HYDRA_THRUST_FWD(f)
, make_index_sequence<sizeof...(Ts)>{}
)
);

template <typename... Ts, typename F>
auto tuple_transform(std::tuple<Ts...>& t, F&& f)
HYDRA_THRUST_DECLTYPE_RETURNS(
detail::tuple_transform_impl(
t
, HYDRA_THRUST_FWD(f)
, make_index_sequence<sizeof...(Ts)>{}
)
);
template <typename... Ts, typename F>
auto tuple_transform(std::tuple<Ts...> const& t, F&& f)
HYDRA_THRUST_DECLTYPE_RETURNS(
detail::tuple_transform_impl(
t
, HYDRA_THRUST_FWD(f)
, make_index_sequence<sizeof...(Ts)>{}
)
);
template <typename... Ts, typename F>
auto tuple_transform(std::tuple<Ts...>&& t, F&& f)
HYDRA_THRUST_DECLTYPE_RETURNS(
detail::tuple_transform_impl(
std::move(t)
, HYDRA_THRUST_FWD(f)
, make_index_sequence<sizeof...(Ts)>{}
)
);

HYDRA_THRUST_END_NS

#endif 

