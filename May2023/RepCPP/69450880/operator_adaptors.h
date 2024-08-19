

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/tuple.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>

namespace hydra_thrust
{
namespace detail
{
namespace functional
{

template<template<typename> class UnaryOperator>
struct unary_operator
{
template<typename Env>
struct argument
: hydra_thrust::detail::eval_if<
(hydra_thrust::tuple_size<Env>::value == 0),
hydra_thrust::detail::identity_<hydra_thrust::null_type>,
hydra_thrust::tuple_element<0,Env>
>
{
};

template<typename Env>
struct operator_type
{
typedef UnaryOperator<
typename hydra_thrust::detail::remove_reference<
typename argument<Env>::type
>::type
> type;
};

template<typename Env>
struct result
{
typedef typename operator_type<Env>::type op_type;
typedef typename op_type::result_type type;
};

template<typename Env>
__host__ __device__
typename result<Env>::type eval(const Env &e) const
{
typename operator_type<Env>::type op;
return op(hydra_thrust::get<0>(e));
} 
}; 

template<template<typename> class BinaryOperator>
struct binary_operator
{
template<typename Env>
struct first_argument
: hydra_thrust::detail::eval_if<
(hydra_thrust::tuple_size<Env>::value == 0),
hydra_thrust::detail::identity_<hydra_thrust::null_type>,
hydra_thrust::tuple_element<0,Env>
>
{
};

template<typename Env>
struct operator_type
{
typedef BinaryOperator<
typename hydra_thrust::detail::remove_reference<
typename first_argument<Env>::type
>::type
> type;
};

template<typename Env>
struct result
{
typedef typename operator_type<Env>::type op_type;
typedef typename op_type::result_type type;
};

template<typename Env>
__host__ __device__
typename result<Env>::type eval(const Env &e) const
{
typename operator_type<Env>::type op;
return op(hydra_thrust::get<0>(e), hydra_thrust::get<1>(e));
} 
}; 

} 
} 
} 

