

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/tuple.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>

namespace hydra_thrust
{
namespace detail
{
namespace tuple_detail
{


template<int i, typename Tuple, typename T>
struct get_or_result
: eval_if<
i < tuple_size<Tuple>::value,
tuple_element<i,Tuple>,
identity_<T>
>
{};


template<int i, typename Tuple, typename T>
inline __host__ __device__
typename lazy_enable_if<
i < tuple_size<Tuple>::value,
tuple_element<i,Tuple>
>::type
get_or(const Tuple &t, const T &val)
{
return hydra_thrust::get<i>(t);
}


template<int i, typename Tuple, typename T>
inline __host__ __device__
typename enable_if<
i >= tuple_size<Tuple>::value,
const T&
>::type
get_or(const Tuple &t, const T &val)
{
return val;
}


template<int i, typename Tuple, typename T>
inline __host__ __device__
typename enable_if<
i >= tuple_size<Tuple>::value,
T&
>::type
get_or(const Tuple &t, T &val)
{
return val;
}


template<int i, int limit, typename Tuple>
struct tuple_element_or_null
: eval_if<
i < limit,
tuple_element<i,Tuple>,
identity_<null_type>
>
{};


template<typename Tuple>
struct tuple_tail_result
{
static const int limit = tuple_size<Tuple>::value;

typedef hydra_thrust::tuple<
typename tuple_element_or_null<1,limit,Tuple>::type,
typename tuple_element_or_null<2,limit,Tuple>::type,
typename tuple_element_or_null<3,limit,Tuple>::type,
typename tuple_element_or_null<4,limit,Tuple>::type,
typename tuple_element_or_null<5,limit,Tuple>::type,
typename tuple_element_or_null<6,limit,Tuple>::type,
typename tuple_element_or_null<7,limit,Tuple>::type,
typename tuple_element_or_null<8,limit,Tuple>::type,
typename tuple_element_or_null<9,limit,Tuple>::type
> type;
};


template<typename Tuple>
inline __host__ __device__
typename enable_if<
(tuple_size<Tuple>::value > 1),
typename tuple_tail_result<Tuple>::type
>::type
tuple_tail(const Tuple &t)
{
typedef typename tuple_tail_result<Tuple>::type result_type;

return result_type(get_or<1>(t, null_type()),
get_or<2>(t, null_type()),
get_or<3>(t, null_type()),
get_or<4>(t, null_type()),
get_or<5>(t, null_type()),
get_or<6>(t, null_type()),
get_or<7>(t, null_type()),
get_or<8>(t, null_type()),
get_or<9>(t, null_type()));
}


template<typename Tuple>
inline __host__ __device__
typename enable_if<
(tuple_size<Tuple>::value <= 1),
typename tuple_tail_result<Tuple>::type
>::type
tuple_tail(const Tuple &t)
{
typedef typename tuple_tail_result<Tuple>::type result_type;


return result_type();
}


template<bool b, typename True, typename False>
inline __host__ __device__
typename enable_if<b, const True&>::type
if_else(const True &t, const False &f)
{
return t;
}


template<bool b, typename True, typename False>
inline __host__ __device__
typename enable_if<b, True&>::type
if_else(True &t, const False &f)
{
return t;
}


template<bool b, typename True, typename False>
inline __host__ __device__
typename enable_if<!b, False&>::type
if_else(const True &t, False &f)
{
return f;
}


template<bool b, typename True, typename False>
inline __host__ __device__
typename enable_if<!b, const False&>::type
if_else(const True &t, const False &f)
{
return f;
}


template<typename T, typename Tuple>
struct tuple_append_result
{
static const int append_slot = hydra_thrust::tuple_size<Tuple>::value;

template<int i>
struct null_unless_append_slot
: eval_if<
i == append_slot,
identity_<T>,
identity_<null_type>
>
{};


typedef hydra_thrust::tuple<
typename get_or_result<0,Tuple, typename null_unless_append_slot<0>::type>::type,
typename get_or_result<1,Tuple, typename null_unless_append_slot<1>::type>::type,
typename get_or_result<2,Tuple, typename null_unless_append_slot<2>::type>::type,
typename get_or_result<3,Tuple, typename null_unless_append_slot<3>::type>::type,
typename get_or_result<4,Tuple, typename null_unless_append_slot<4>::type>::type,
typename get_or_result<5,Tuple, typename null_unless_append_slot<5>::type>::type,
typename get_or_result<6,Tuple, typename null_unless_append_slot<6>::type>::type,
typename get_or_result<7,Tuple, typename null_unless_append_slot<7>::type>::type,
typename get_or_result<8,Tuple, typename null_unless_append_slot<8>::type>::type,
typename get_or_result<9,Tuple, typename null_unless_append_slot<9>::type>::type
> type;
};


template<typename T, typename Tuple>
inline __host__ __device__
typename enable_if<
tuple_size<Tuple>::value < 10,
typename tuple_append_result<T,Tuple>::type
>::type
tuple_append(const Tuple &t, const T &x)
{
const int x_slot = hydra_thrust::tuple_size<Tuple>::value;

typedef typename tuple_append_result<T,Tuple>::type result_type;


return result_type(get_or<0>(t,if_else<0 == x_slot>(x, null_type())),
get_or<1>(t,if_else<1 == x_slot>(x, null_type())),
get_or<2>(t,if_else<2 == x_slot>(x, null_type())),
get_or<3>(t,if_else<3 == x_slot>(x, null_type())),
get_or<4>(t,if_else<4 == x_slot>(x, null_type())),
get_or<5>(t,if_else<5 == x_slot>(x, null_type())),
get_or<6>(t,if_else<6 == x_slot>(x, null_type())),
get_or<7>(t,if_else<7 == x_slot>(x, null_type())),
get_or<8>(t,if_else<8 == x_slot>(x, null_type())),
get_or<9>(t,if_else<9 == x_slot>(x, null_type())));
}


template<typename T, typename Tuple>
inline __host__ __device__
typename enable_if<
tuple_size<Tuple>::value < 10,
typename tuple_append_result<T,Tuple>::type
>::type
tuple_append(const Tuple &t, T &x)
{
static const int x_slot = hydra_thrust::tuple_size<Tuple>::value;

typedef typename tuple_append_result<T,Tuple>::type result_type;


return result_type(get_or<0>(t,if_else<0 == x_slot>(x, null_type())),
get_or<1>(t,if_else<1 == x_slot>(x, null_type())),
get_or<2>(t,if_else<2 == x_slot>(x, null_type())),
get_or<3>(t,if_else<3 == x_slot>(x, null_type())),
get_or<4>(t,if_else<4 == x_slot>(x, null_type())),
get_or<5>(t,if_else<5 == x_slot>(x, null_type())),
get_or<6>(t,if_else<6 == x_slot>(x, null_type())),
get_or<7>(t,if_else<7 == x_slot>(x, null_type())),
get_or<8>(t,if_else<8 == x_slot>(x, null_type())),
get_or<9>(t,if_else<9 == x_slot>(x, null_type())));
}


} 
} 
} 

