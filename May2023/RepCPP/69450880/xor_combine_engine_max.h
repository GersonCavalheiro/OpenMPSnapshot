

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/mpl/math.h>
#include <limits>
#include <cstddef>

namespace hydra_thrust
{

namespace random
{

namespace detail
{


namespace math = hydra_thrust::detail::mpl::math;


namespace detail
{

template<typename UIntType, UIntType w,
UIntType lhs, UIntType rhs,
bool shift_will_overflow>
struct lshift_w
{
static const UIntType value = 0;
};


template<typename UIntType, UIntType w,
UIntType lhs, UIntType rhs>
struct lshift_w<UIntType,w,lhs,rhs,false>
{
static const UIntType value = lhs << rhs;
};

} 


template<typename UIntType, UIntType w,
UIntType lhs, UIntType rhs>
struct lshift_w
{
static const bool shift_will_overflow = rhs >= w;

static const UIntType value = detail::lshift_w<UIntType, w, lhs, rhs, shift_will_overflow>::value;
};


template<typename UIntType, UIntType lhs, UIntType rhs>
struct lshift
: lshift_w<UIntType, std::numeric_limits<UIntType>::digits, lhs, rhs>
{};


template<typename UIntType, int p>
struct two_to_the_power
: lshift<UIntType, 1, p>
{};


template<typename result_type, result_type a, result_type b, int d>
class xor_combine_engine_max_aux_constants
{
public:
static const result_type two_to_the_d = two_to_the_power<result_type, d>::value;
static const result_type c = lshift<result_type, a, d>::value;

static const result_type t =
math::max<
result_type,
c,
b
>::value;

static const result_type u =
math::min<
result_type,
c,
b
>::value;

static const result_type p            = math::log2<u>::value;
static const result_type two_to_the_p = two_to_the_power<result_type, p>::value;

static const result_type k = math::div<result_type, t, two_to_the_p>::value;
};


template<typename result_type, result_type, result_type, int> struct xor_combine_engine_max_aux;


template<typename result_type, result_type a, result_type b, int d>
struct xor_combine_engine_max_aux_case4
{
typedef xor_combine_engine_max_aux_constants<result_type,a,b,d> constants;

static const result_type k_plus_1_times_two_to_the_p =
lshift<
result_type,
math::plus<result_type,constants::k,1>::value,
constants::p
>::value;

static const result_type M =
xor_combine_engine_max_aux<
result_type,
math::div<
result_type,
math::mod<
result_type,
constants::u,
constants::two_to_the_p
>::value,
constants::two_to_the_p
>::value,
math::mod<
result_type,
constants::t,
constants::two_to_the_p
>::value,
d
>::value;

static const result_type value = math::plus<result_type, k_plus_1_times_two_to_the_p, M>::value;
};


template<typename result_type, result_type a, result_type b, int d>
struct xor_combine_engine_max_aux_case3
{
typedef xor_combine_engine_max_aux_constants<result_type,a,b,d> constants;

static const result_type k_plus_1_times_two_to_the_p =
lshift<
result_type,
math::plus<result_type,constants::k,1>::value,
constants::p
>::value;

static const result_type M =
xor_combine_engine_max_aux<
result_type,
math::div<
result_type,
math::mod<
result_type,
constants::t,
constants::two_to_the_p
>::value,
constants::two_to_the_p
>::value,
math::mod<
result_type,
constants::u,
constants::two_to_the_p
>::value,
d
>::value;

static const result_type value = math::plus<result_type, k_plus_1_times_two_to_the_p, M>::value;
};



template<typename result_type, result_type a, result_type b, int d>
struct xor_combine_engine_max_aux_case2
{
typedef xor_combine_engine_max_aux_constants<result_type,a,b,d> constants;

static const result_type k_plus_1_times_two_to_the_p =
lshift<
result_type,
math::plus<result_type,constants::k,1>::value,
constants::p
>::value;

static const result_type value =
math::minus<
result_type,
k_plus_1_times_two_to_the_p,
1
>::value;
};


template<typename result_type, result_type a, result_type b, int d>
struct xor_combine_engine_max_aux_case1
{
static const result_type c     = lshift<result_type, a, d>::value;

static const result_type value = math::plus<result_type,c,b>::value;
};


template<typename result_type, result_type a, result_type b, int d>
struct xor_combine_engine_max_aux_2
{
typedef xor_combine_engine_max_aux_constants<result_type,a,b,d> constants;

static const result_type value = 
hydra_thrust::detail::eval_if<
math::is_odd<result_type, constants::k>::value,
hydra_thrust::detail::identity_<
hydra_thrust::detail::integral_constant<
result_type,
xor_combine_engine_max_aux_case2<result_type,a,b,d>::value
>
>,
hydra_thrust::detail::eval_if<
a * constants::two_to_the_d >= b,
hydra_thrust::detail::identity_<
hydra_thrust::detail::integral_constant<
result_type,
xor_combine_engine_max_aux_case3<result_type,a,b,d>::value
>
>,
hydra_thrust::detail::identity_<
hydra_thrust::detail::integral_constant<
result_type,
xor_combine_engine_max_aux_case4<result_type,a,b,d>::value
>
>
>
>::type::value;
};


template<typename result_type,
result_type a,
result_type b,
int d,
bool use_case1 = (a == 0) || (b < two_to_the_power<result_type,d>::value)>
struct xor_combine_engine_max_aux_1
: xor_combine_engine_max_aux_case1<result_type,a,b,d>
{};


template<typename result_type,
result_type a,
result_type b,
int d>
struct xor_combine_engine_max_aux_1<result_type,a,b,d,false>
: xor_combine_engine_max_aux_2<result_type,a,b,d>
{};


template<typename result_type,
result_type a,
result_type b,
int d>
struct xor_combine_engine_max_aux
: xor_combine_engine_max_aux_1<result_type,a,b,d>
{};


template<typename Engine1, size_t s1, typename Engine2, size_t s2, typename result_type>
struct xor_combine_engine_max
{
static const size_t w = std::numeric_limits<result_type>::digits;

static const result_type m1 =
math::min<
result_type,
result_type(Engine1::max - Engine1::min),
two_to_the_power<result_type, w-s1>::value - 1 
>::value;

static const result_type m2 =
math::min<
result_type,
result_type(Engine2::max - Engine2::min),
two_to_the_power<result_type, w-s2>::value - 1
>::value;

static const result_type s = s1 - s2;

static const result_type M =
xor_combine_engine_max_aux<
result_type,
m1,
m2,
s
>::value;

static const result_type value =
lshift_w<
result_type,
w,
M,
s2
>::value;
}; 

} 

} 

} 

