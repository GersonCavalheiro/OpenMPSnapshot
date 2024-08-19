

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
binary_operator<hydra_thrust::equal_to>,
actor<T1>,
typename as_actor<T2>::type
>
>
operator==(const actor<T1> &_1, const T2 &_2)
{
return compose(binary_operator<hydra_thrust::equal_to>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<hydra_thrust::equal_to>,
typename as_actor<T1>::type,
actor<T2>
>
>
operator==(const T1 &_1, const actor<T2> &_2)
{
return compose(binary_operator<hydra_thrust::equal_to>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<hydra_thrust::equal_to>,
actor<T1>,
actor<T2>
>
>
operator==(const actor<T1> &_1, const actor<T2> &_2)
{
return compose(binary_operator<hydra_thrust::equal_to>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<hydra_thrust::not_equal_to>,
actor<T1>,
typename as_actor<T2>::type
>
>
operator!=(const actor<T1> &_1, const T2 &_2)
{
return compose(binary_operator<hydra_thrust::not_equal_to>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<hydra_thrust::not_equal_to>,
typename as_actor<T1>::type,
actor<T2>
>
>
operator!=(const T1 &_1, const actor<T2> &_2)
{
return compose(binary_operator<hydra_thrust::not_equal_to>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<hydra_thrust::not_equal_to>,
actor<T1>,
actor<T2>
>
>
operator!=(const actor<T1> &_1, const actor<T2> &_2)
{
return compose(binary_operator<hydra_thrust::not_equal_to>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<hydra_thrust::greater>,
actor<T1>,
typename as_actor<T2>::type
>
>
operator>(const actor<T1> &_1, const T2 &_2)
{
return compose(binary_operator<hydra_thrust::greater>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<hydra_thrust::greater>,
typename as_actor<T1>::type,
actor<T2>
>
>
operator>(const T1 &_1, const actor<T2> &_2)
{
return compose(binary_operator<hydra_thrust::greater>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<hydra_thrust::greater>,
actor<T1>,
actor<T2>
>
>
operator>(const actor<T1> &_1, const actor<T2> &_2)
{
return compose(binary_operator<hydra_thrust::greater>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<hydra_thrust::less>,
actor<T1>,
typename as_actor<T2>::type
>
>
operator<(const actor<T1> &_1, const T2 &_2)
{
return compose(binary_operator<hydra_thrust::less>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<hydra_thrust::less>,
typename as_actor<T1>::type,
actor<T2>
>
>
operator<(const T1 &_1, const actor<T2> &_2)
{
return compose(binary_operator<hydra_thrust::less>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<hydra_thrust::less>,
actor<T1>,
actor<T2>
>
>
operator<(const actor<T1> &_1, const actor<T2> &_2)
{
return compose(binary_operator<hydra_thrust::less>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<hydra_thrust::greater_equal>,
actor<T1>,
typename as_actor<T2>::type
>
>
operator>=(const actor<T1> &_1, const T2 &_2)
{
return compose(binary_operator<hydra_thrust::greater_equal>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<hydra_thrust::greater_equal>,
typename as_actor<T1>::type,
actor<T2>
>
>
operator>=(const T1 &_1, const actor<T2> &_2)
{
return compose(binary_operator<hydra_thrust::greater_equal>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<hydra_thrust::greater_equal>,
actor<T1>,
actor<T2>
>
>
operator>=(const actor<T1> &_1, const actor<T2> &_2)
{
return compose(binary_operator<hydra_thrust::greater_equal>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<hydra_thrust::less_equal>,
actor<T1>,
typename as_actor<T2>::type
>
>
operator<=(const actor<T1> &_1, const T2 &_2)
{
return compose(binary_operator<hydra_thrust::less_equal>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<hydra_thrust::less_equal>,
typename as_actor<T1>::type,
actor<T2>
>
>
operator<=(const T1 &_1, const actor<T2> &_2)
{
return compose(binary_operator<hydra_thrust::less_equal>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<hydra_thrust::less_equal>,
actor<T1>,
actor<T2>
>
>
operator<=(const actor<T1> &_1, const actor<T2> &_2)
{
return compose(binary_operator<hydra_thrust::less_equal>(),
make_actor(_1),
make_actor(_2));
} 

} 
} 
} 

