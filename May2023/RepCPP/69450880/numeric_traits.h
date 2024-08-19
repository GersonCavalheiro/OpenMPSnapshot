

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <limits>


namespace hydra_thrust
{

namespace detail
{

typedef long long intmax_t;

template<typename Number>
struct is_signed
: integral_constant<bool, std::numeric_limits<Number>::is_signed>
{}; 


template<typename T>
struct num_digits
: eval_if<
std::numeric_limits<T>::is_specialized,
integral_constant<
int,
std::numeric_limits<T>::digits
>,
integral_constant<
int,
sizeof(T) * std::numeric_limits<unsigned char>::digits - (is_signed<T>::value ? 1 : 0)  
>
>::type
{}; 


template<typename Integer>
struct integer_difference
{
private:
template<bool x, bool y>
struct and_
{
static const bool value = false;
};

template<bool y>
struct and_<true,y>
{
static const bool value = y;
};

public:
typedef typename
eval_if<
and_<
std::numeric_limits<Integer>::is_signed,
(!std::numeric_limits<Integer>::is_bounded || (int(std::numeric_limits<Integer>::digits) + 1 >= num_digits<intmax_t>::value))
>::value,
identity_<Integer>,
eval_if<
int(std::numeric_limits<Integer>::digits) + 1 < num_digits<signed int>::value,
identity_<signed int>,
eval_if<
int(std::numeric_limits<Integer>::digits) + 1 < num_digits<signed long>::value,
identity_<signed long>,
identity_<intmax_t>
>
>
>::type type;
}; 


template<typename Number>
struct numeric_difference
: eval_if<
is_integral<Number>::value,
integer_difference<Number>,
identity_<Number>
>
{}; 


template<typename Number>
__host__ __device__
typename numeric_difference<Number>::type
numeric_distance(Number x, Number y)
{
typedef typename numeric_difference<Number>::type difference_type;
return difference_type(y) - difference_type(x);
} 

} 

} 

