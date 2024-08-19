

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
binary_operator<hydra_thrust::bit_and>,
actor<T1>,
typename as_actor<T2>::type
>
>
operator&(const actor<T1> &_1, const T2 &_2)
{
return compose(binary_operator<hydra_thrust::bit_and>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<hydra_thrust::bit_and>,
typename as_actor<T1>::type,
actor<T2>
>
>
operator&(const T1 &_1, const actor<T2> &_2)
{
return compose(binary_operator<hydra_thrust::bit_and>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<hydra_thrust::bit_and>,
actor<T1>,
actor<T2>
>
>
operator&(const actor<T1> &_1, const actor<T2> &_2)
{
return compose(binary_operator<hydra_thrust::bit_and>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<hydra_thrust::bit_or>,
actor<T1>,
typename as_actor<T2>::type
>
>
operator|(const actor<T1> &_1, const T2 &_2)
{
return compose(binary_operator<hydra_thrust::bit_or>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<hydra_thrust::bit_or>,
typename as_actor<T1>::type,
actor<T2>
>
>
operator|(const T1 &_1, const actor<T2> &_2)
{
return compose(binary_operator<hydra_thrust::bit_or>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<hydra_thrust::bit_or>,
actor<T1>,
actor<T2>
>
>
operator|(const actor<T1> &_1, const actor<T2> &_2)
{
return compose(binary_operator<hydra_thrust::bit_or>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<hydra_thrust::bit_xor>,
actor<T1>,
typename as_actor<T2>::type
>
>
operator^(const actor<T1> &_1, const T2 &_2)
{
return compose(binary_operator<hydra_thrust::bit_xor>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<hydra_thrust::bit_xor>,
typename as_actor<T1>::type,
actor<T2>
>
>
operator^(const T1 &_1, const actor<T2> &_2)
{
return compose(binary_operator<hydra_thrust::bit_xor>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<hydra_thrust::bit_xor>,
actor<T1>,
actor<T2>
>
>
operator^(const actor<T1> &_1, const actor<T2> &_2)
{
return compose(binary_operator<hydra_thrust::bit_xor>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T>
struct bit_not
: public hydra_thrust::unary_function<T,T>
{
__host__ __device__ T operator()(const T &x) const {return ~x;}
}; 

template<typename Eval>
__host__ __device__
actor<
composite<
unary_operator<bit_not>,
actor<Eval>
>
>
__host__ __device__
operator~(const actor<Eval> &_1)
{
return compose(unary_operator<bit_not>(), _1);
} 

template<typename T>
struct bit_lshift
: public hydra_thrust::binary_function<T,T,T>
{
__host__ __device__ T operator()(const T &lhs, const T &rhs) const {return lhs << rhs;}
}; 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<bit_lshift>,
actor<T1>,
typename as_actor<T2>::type
>
>
operator<<(const actor<T1> &_1, const T2 &_2)
{
return compose(binary_operator<bit_lshift>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<bit_lshift>,
typename as_actor<T1>::type,
actor<T2>
>
>
operator<<(const T1 &_1, const actor<T2> &_2)
{
return compose(binary_operator<bit_lshift>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<bit_lshift>,
actor<T1>,
actor<T2>
>
>
operator<<(const actor<T1> &_1, const actor<T2> &_2)
{
return compose(binary_operator<bit_lshift>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T>
struct bit_rshift
: public hydra_thrust::binary_function<T,T,T>
{
__host__ __device__ T operator()(const T &lhs, const T &rhs) const {return lhs >> rhs;}
}; 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<bit_rshift>,
actor<T1>,
typename as_actor<T2>::type
>
>
operator>>(const actor<T1> &_1, const T2 &_2)
{
return compose(binary_operator<bit_rshift>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<bit_rshift>,
typename as_actor<T1>::type,
actor<T2>
>
>
operator>>(const T1 &_1, const actor<T2> &_2)
{
return compose(binary_operator<bit_rshift>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<bit_rshift>,
actor<T1>,
actor<T2>
>
>
operator>>(const actor<T1> &_1, const actor<T2> &_2)
{
return compose(binary_operator<bit_rshift>(),
make_actor(_1),
make_actor(_2));
} 

} 
} 
} 

