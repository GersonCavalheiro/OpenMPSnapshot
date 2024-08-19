

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/functional/actor.h>
#include <hydra/detail/external/hydra_thrust/detail/functional/composite.h>
#include <hydra/detail/external/hydra_thrust/detail/functional/operators/operator_adaptors.h>

namespace hydra_thrust
{
namespace detail
{
namespace functional
{

template<typename T>
struct plus_equal
: public hydra_thrust::binary_function<T&,T,T&>
{
__host__ __device__ T& operator()(T &lhs, const T &rhs) const { return lhs += rhs; }
}; 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<plus_equal>,
actor<T1>,
typename as_actor<T2>::type
>
>
operator+=(const actor<T1> &_1, const T2 &_2)
{
return compose(binary_operator<plus_equal>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<plus_equal>,
actor<T1>,
actor<T2>
>
>
operator+=(const actor<T1> &_1, const actor<T2> &_2)
{
return compose(binary_operator<plus_equal>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T>
struct minus_equal
: public hydra_thrust::binary_function<T&,T,T&>
{
__host__ __device__ T& operator()(T &lhs, const T &rhs) const { return lhs -= rhs; }
}; 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<minus_equal>,
actor<T1>,
typename as_actor<T2>::type
>
>
operator-=(const actor<T1> &_1, const T2 &_2)
{
return compose(binary_operator<minus_equal>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<minus_equal>,
actor<T1>,
actor<T2>
>
>
operator-=(const actor<T1> &_1, const actor<T2> &_2)
{
return compose(binary_operator<minus_equal>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T>
struct multiplies_equal
: public hydra_thrust::binary_function<T&,T,T&>
{
__host__ __device__ T& operator()(T &lhs, const T&rhs) const { return lhs *= rhs; }
}; 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<multiplies_equal>,
actor<T1>,
typename as_actor<T2>::type
>
>
operator*=(const actor<T1> &_1, const T2 &_2)
{
return compose(binary_operator<multiplies_equal>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<multiplies_equal>,
actor<T1>,
actor<T2>
>
>
operator*=(const actor<T1> &_1, const actor<T2> &_2)
{
return compose(binary_operator<multiplies_equal>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T>
struct divides_equal
: public hydra_thrust::binary_function<T&,T,T&>
{
__host__ __device__ T& operator()(T &lhs, const T&rhs) const { return lhs /= rhs; }
}; 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<divides_equal>,
actor<T1>,
typename as_actor<T2>::type
>
>
operator/=(const actor<T1> &_1, const T2 &_2)
{
return compose(binary_operator<divides_equal>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<divides_equal>,
actor<T1>,
actor<T2>
>
>
operator/=(const actor<T1> &_1, const actor<T2> &_2)
{
return compose(binary_operator<divides_equal>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T>
struct modulus_equal
: public hydra_thrust::binary_function<T&,T,T&>
{
__host__ __device__ T& operator()(T &lhs, const T&rhs) const { return lhs %= rhs; }
}; 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<modulus_equal>,
actor<T1>,
typename as_actor<T2>::type
>
>
operator%=(const actor<T1> &_1, const T2 &_2)
{
return compose(binary_operator<modulus_equal>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<modulus_equal>,
actor<T1>,
actor<T2>
>
>
operator%=(const actor<T1> &_1, const actor<T2> &_2)
{
return compose(binary_operator<modulus_equal>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T>
struct bit_and_equal
: public hydra_thrust::binary_function<T&,T,T&>
{
__host__ __device__ T& operator()(T &lhs, const T&rhs) const { return lhs &= rhs; }
}; 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<bit_and_equal>,
actor<T1>,
typename as_actor<T2>::type
>
>
operator&=(const actor<T1> &_1, const T2 &_2)
{
return compose(binary_operator<bit_and_equal>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<bit_and_equal>,
actor<T1>,
actor<T2>
>
>
operator&=(const actor<T1> &_1, const actor<T2> &_2)
{
return compose(binary_operator<bit_and_equal>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T>
struct bit_or_equal
: public hydra_thrust::binary_function<T&,T,T&>
{
__host__ __device__ T& operator()(T &lhs, const T&rhs) const { return lhs |= rhs; }
}; 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<bit_or_equal>,
actor<T1>,
typename as_actor<T2>::type
>
>
operator|=(const actor<T1> &_1, const T2 &_2)
{
return compose(binary_operator<bit_or_equal>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<bit_or_equal>,
actor<T1>,
actor<T2>
>
>
operator|=(const actor<T1> &_1, const actor<T2> &_2)
{
return compose(binary_operator<bit_or_equal>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T>
struct bit_xor_equal
: public hydra_thrust::binary_function<T&,T,T&>
{
__host__ __device__ T& operator()(T &lhs, const T&rhs) const { return lhs ^= rhs; }
}; 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<bit_xor_equal>,
actor<T1>,
typename as_actor<T2>::type
>
>
operator^=(const actor<T1> &_1, const T2 &_2)
{
return compose(binary_operator<bit_xor_equal>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<bit_xor_equal>,
actor<T1>,
actor<T2>
>
>
operator^=(const actor<T1> &_1, const actor<T2> &_2)
{
return compose(binary_operator<bit_xor_equal>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T>
struct bit_lshift_equal
: public hydra_thrust::binary_function<T&,T,T&>
{
__host__ __device__ T& operator()(T &lhs, const T&rhs) const { return lhs <<= rhs; }
}; 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<bit_lshift_equal>,
actor<T1>,
typename as_actor<T2>::type
>
>
operator<<=(const actor<T1> &_1, const T2 &_2)
{
return compose(binary_operator<bit_lshift_equal>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<bit_lshift_equal>,
actor<T1>,
actor<T2>
>
>
operator<<=(const actor<T1> &_1, const actor<T2> &_2)
{
return compose(binary_operator<bit_lshift_equal>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T>
struct bit_rshift_equal
: public hydra_thrust::binary_function<T&,T,T&>
{
__host__ __device__ T& operator()(T &lhs, const T&rhs) const { return lhs >>= rhs; }
}; 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<bit_rshift_equal>,
actor<T1>,
typename as_actor<T2>::type
>
>
operator>>=(const actor<T1> &_1, const T2 &_2)
{
return compose(binary_operator<bit_rshift_equal>(),
make_actor(_1),
make_actor(_2));
} 

template<typename T1, typename T2>
__host__ __device__
actor<
composite<
binary_operator<bit_rshift_equal>,
actor<T1>,
actor<T2>
>
>
operator>>=(const actor<T1> &_1, const actor<T2> &_2)
{
return compose(binary_operator<bit_rshift_equal>(),
make_actor(_1),
make_actor(_2));
} 

} 
} 
} 

