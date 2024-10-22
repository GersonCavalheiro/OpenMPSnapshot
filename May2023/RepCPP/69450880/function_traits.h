

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits/has_nested_type.h>

namespace hydra_thrust
{

template <typename T> struct plus;
template <typename T> struct multiplies;
template <typename T> struct minimum;
template <typename T> struct maximum;
template <typename T> struct logical_or;
template <typename T> struct logical_and;
template <typename T> struct bit_or;
template <typename T> struct bit_and;
template <typename T> struct bit_xor;

namespace detail
{



__HYDRA_THRUST_DEFINE_HAS_NESTED_TYPE(has_result_type, result_type)

__HYDRA_THRUST_DEFINE_HAS_NESTED_TYPE(has_argument_type, argument_type)

__HYDRA_THRUST_DEFINE_HAS_NESTED_TYPE(has_first_argument_type, first_argument_type)

__HYDRA_THRUST_DEFINE_HAS_NESTED_TYPE(has_second_argument_type, second_argument_type)


template<typename AdaptableBinaryFunction>
struct result_type
{
typedef typename AdaptableBinaryFunction::result_type type;
};


template<typename T>
struct is_adaptable_unary_function
: hydra_thrust::detail::and_<
has_result_type<T>,
has_argument_type<T>
>
{};


template<typename T>
struct is_adaptable_binary_function
: hydra_thrust::detail::and_<
has_result_type<T>,
hydra_thrust::detail::and_<
has_first_argument_type<T>,
has_second_argument_type<T>
>
>
{};


template<typename BinaryFunction>
struct is_commutative
: public hydra_thrust::detail::false_type
{};

template<typename T> struct is_commutative< typename hydra_thrust::plus<T>        > : public hydra_thrust::detail::is_arithmetic<T> {};
template<typename T> struct is_commutative< typename hydra_thrust::multiplies<T>  > : public hydra_thrust::detail::is_arithmetic<T> {};
template<typename T> struct is_commutative< typename hydra_thrust::minimum<T>     > : public hydra_thrust::detail::is_arithmetic<T> {};
template<typename T> struct is_commutative< typename hydra_thrust::maximum<T>     > : public hydra_thrust::detail::is_arithmetic<T> {};
template<typename T> struct is_commutative< typename hydra_thrust::logical_or<T>  > : public hydra_thrust::detail::is_arithmetic<T> {};
template<typename T> struct is_commutative< typename hydra_thrust::logical_and<T> > : public hydra_thrust::detail::is_arithmetic<T> {};
template<typename T> struct is_commutative< typename hydra_thrust::bit_or<T>      > : public hydra_thrust::detail::is_arithmetic<T> {};
template<typename T> struct is_commutative< typename hydra_thrust::bit_and<T>     > : public hydra_thrust::detail::is_arithmetic<T> {};
template<typename T> struct is_commutative< typename hydra_thrust::bit_xor<T>     > : public hydra_thrust::detail::is_arithmetic<T> {};

} 
} 

