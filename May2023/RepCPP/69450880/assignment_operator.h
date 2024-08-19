

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/functional/actor.h>
#include <hydra/detail/external/hydra_thrust/detail/functional/composite.h>
#include <hydra/detail/external/hydra_thrust/detail/functional/operators/operator_adaptors.h>
#include <hydra/detail/external/hydra_thrust/functional.h>

namespace hydra_thrust
{

template<typename,typename,typename> struct binary_function;

namespace detail
{
namespace functional
{

template<typename> struct as_actor;

template<typename T>
struct assign
: hydra_thrust::binary_function<T&,T,T&>
{
__host__ __device__ T& operator()(T &lhs, const T &rhs) const { return lhs = rhs; }
}; 

template<typename Eval, typename T>
struct assign_result
{
typedef actor<
composite<
binary_operator<assign>,
actor<Eval>,
typename as_actor<T>::type
>
> type;
}; 

template<typename Eval, typename T>
__host__ __device__
typename assign_result<Eval,T>::type
do_assign(const actor<Eval> &_1, const T &_2)
{
return compose(binary_operator<assign>(),
_1,
as_actor<T>::convert(_2));
} 

} 
} 
} 

