


#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/tuple.h>
#include <hydra/detail/external/hydra_thrust/detail/functional/value.h>
#include <hydra/detail/external/hydra_thrust/detail/functional/composite.h>
#include <hydra/detail/external/hydra_thrust/detail/functional/operators/assignment_operator.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits/result_of_adaptable_function.h>

namespace hydra_thrust
{
namespace detail
{
namespace functional
{

template<typename Action, typename Env>
struct apply_actor
{
typedef typename Action::template result<Env>::type type;
};

template<typename Eval>
struct actor
: Eval
{
typedef Eval eval_type;

__host__ __device__
actor(void);

__host__ __device__
actor(const Eval &base);

__host__ __device__
typename apply_actor<eval_type, hydra_thrust::null_type >::type
operator()(void) const;

template<typename T0>
__host__ __device__
typename apply_actor<eval_type, hydra_thrust::tuple<T0&> >::type
operator()(T0 &_0) const;

template<typename T0, typename T1>
__host__ __device__
typename apply_actor<eval_type, hydra_thrust::tuple<T0&,T1&> >::type
operator()(T0 &_0, T1 &_1) const;

template<typename T0, typename T1, typename T2>
__host__ __device__
typename apply_actor<eval_type, hydra_thrust::tuple<T0&,T1&,T2&> >::type
operator()(T0 &_0, T1 &_1, T2 &_2) const;

template<typename T0, typename T1, typename T2, typename T3>
__host__ __device__
typename apply_actor<eval_type, hydra_thrust::tuple<T0&,T1&,T2&,T3&> >::type
operator()(T0 &_0, T1 &_1, T2 &_2, T3 &_3) const;

template<typename T0, typename T1, typename T2, typename T3, typename T4>
__host__ __device__
typename apply_actor<eval_type, hydra_thrust::tuple<T0&,T1&,T2&,T3&,T4&> >::type
operator()(T0 &_0, T1 &_1, T2 &_2, T3 &_3, T4 &_4) const;

template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5>
__host__ __device__
typename apply_actor<eval_type, hydra_thrust::tuple<T0&,T1&,T2&,T3&,T4&,T5&> >::type
operator()(T0 &_0, T1 &_1, T2 &_2, T3 &_3, T4 &_4, T5 &_5) const;

template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
__host__ __device__
typename apply_actor<eval_type, hydra_thrust::tuple<T0&,T1&,T2&,T3&,T4&,T5&,T6&> >::type
operator()(T0 &_0, T1 &_1, T2 &_2, T3 &_3, T4 &_4, T5 &_5, T6 &_6) const;

template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
__host__ __device__
typename apply_actor<eval_type, hydra_thrust::tuple<T0&,T1&,T2&,T3&,T4&,T5&,T6&,T7&> >::type
operator()(T0 &_0, T1 &_1, T2 &_2, T3 &_3, T4 &_4, T5 &_5, T6 &_6, T7 &_7) const;

template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8>
__host__ __device__
typename apply_actor<eval_type, hydra_thrust::tuple<T0&,T1&,T2&,T3&,T4&,T5&,T6&,T7&,T8&> >::type
operator()(T0 &_0, T1 &_1, T2 &_2, T3 &_3, T4 &_4, T5 &_5, T6 &_6, T7 &_7, T8 &_8) const;

template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9>
__host__ __device__
typename apply_actor<eval_type, hydra_thrust::tuple<T0&,T1&,T2&,T3&,T4&,T5&,T6&,T7&,T8&,T9&> >::type
operator()(T0 &_0, T1 &_1, T2 &_2, T3 &_3, T4 &_4, T5 &_5, T6 &_6, T7 &_7, T8 &_8, T9 &_9) const;

template<typename T>
__host__ __device__
typename assign_result<Eval,T>::type
operator=(const T &_1) const;
}; 

template<typename T>
struct as_actor
{
typedef value<T> type;

static inline __host__ __device__ type convert(const T &x)
{
return val(x);
} 
}; 

template<typename Eval>
struct as_actor<actor<Eval> >
{
typedef actor<Eval> type;

static inline __host__ __device__ const type &convert(const actor<Eval> &x)
{
return x;
} 
}; 

template<typename T>
typename as_actor<T>::type
__host__ __device__
make_actor(const T &x)
{
return as_actor<T>::convert(x);
} 

} 

template<typename Eval>
struct result_of_adaptable_function<
hydra_thrust::detail::functional::actor<Eval>()
>
{
typedef typename hydra_thrust::detail::functional::apply_actor<
hydra_thrust::detail::functional::actor<Eval>,
hydra_thrust::null_type
>::type type;
}; 

template<typename Eval, typename Arg1>
struct result_of_adaptable_function<
hydra_thrust::detail::functional::actor<Eval>(Arg1)
>
{
typedef typename hydra_thrust::detail::functional::apply_actor<
hydra_thrust::detail::functional::actor<Eval>,
hydra_thrust::tuple<Arg1>
>::type type;
}; 

template<typename Eval, typename Arg1, typename Arg2>
struct result_of_adaptable_function<
hydra_thrust::detail::functional::actor<Eval>(Arg1,Arg2)
>
{
typedef typename hydra_thrust::detail::functional::apply_actor<
hydra_thrust::detail::functional::actor<Eval>,
hydra_thrust::tuple<Arg1,Arg2>
>::type type;
}; 

} 
} 

#include <hydra/detail/external/hydra_thrust/detail/functional/actor.inl>

