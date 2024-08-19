

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/functional/actor.h>
#include <hydra/detail/external/hydra_thrust/detail/functional/composite.h>
#include <hydra/detail/external/hydra_thrust/detail/functional/operators/operator_adaptors.h>
#include <hydra/detail/external/hydra_thrust/functional.h>

namespace hydra_thrust
{
namespace detail
{
namespace functional
{

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<hydra_thrust::logical_and>,
actor<T1>,
typename as_actor<T2>::type
>
>
operator&&(const actor<T1> &_1, const T2 &_2)
{
return compose(binary_operator<hydra_thrust::logical_and>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<hydra_thrust::logical_and>,
typename as_actor<T1>::type,
actor<T2>
>
>
operator&&(const T1 &_1, const actor<T2> &_2)
{
return compose(binary_operator<hydra_thrust::logical_and>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<hydra_thrust::logical_and>,
actor<T1>,
actor<T2>
>
>
operator&&(const actor<T1> &_1, const actor<T2> &_2)
{
return compose(binary_operator<hydra_thrust::logical_and>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<hydra_thrust::logical_or>,
actor<T1>,
typename as_actor<T2>::type
>
>
operator||(const actor<T1> &_1, const T2 &_2)
{
return compose(binary_operator<hydra_thrust::logical_or>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<hydra_thrust::logical_or>,
typename as_actor<T1>::type,
actor<T2>
>
>
operator||(const T1 &_1, const actor<T2> &_2)
{
return compose(binary_operator<hydra_thrust::logical_or>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<hydra_thrust::logical_or>,
actor<T1>,
actor<T2>
>
>
operator||(const actor<T1> &_1, const actor<T2> &_2)
{
return compose(binary_operator<hydra_thrust::logical_or>(),
make_actor(_1),
make_actor(_2));
} 

template<typename Eval>
__host__ __device__
actor<
composite<
unary_operator<hydra_thrust::logical_not>,
actor<Eval>
>
>
operator!(const actor<Eval> &_1)
{
return compose(unary_operator<hydra_thrust::logical_not>(), _1);
} 

} 
} 
} 

