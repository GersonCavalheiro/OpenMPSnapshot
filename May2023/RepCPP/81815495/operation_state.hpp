
#ifndef ASIO_EXECUTION_OPERATION_STATE_HPP
#define ASIO_EXECUTION_OPERATION_STATE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/execution/start.hpp"

#if defined(ASIO_HAS_DEDUCED_START_FREE_TRAIT) \
&& defined(ASIO_HAS_DEDUCED_START_MEMBER_TRAIT)
# define ASIO_HAS_DEDUCED_EXECUTION_IS_OPERATION_STATE_TRAIT 1
#endif 

#include "asio/detail/push_options.hpp"

namespace asio {
namespace execution {
namespace detail {

template <typename T>
struct is_operation_state_base :
integral_constant<bool,
is_destructible<T>::value
&& is_object<T>::value
>
{
};

} 


template <typename T>
struct is_operation_state :
#if defined(GENERATING_DOCUMENTATION)
integral_constant<bool, automatically_determined>
#else 
conditional<
can_start<typename add_lvalue_reference<T>::type>::value
&& is_nothrow_start<typename add_lvalue_reference<T>::type>::value,
detail::is_operation_state_base<T>,
false_type
>::type
#endif 
{
};

#if defined(ASIO_HAS_VARIABLE_TEMPLATES)

template <typename T>
ASIO_CONSTEXPR const bool is_operation_state_v =
is_operation_state<T>::value;

#endif 

#if defined(ASIO_HAS_CONCEPTS)

template <typename T>
ASIO_CONCEPT operation_state = is_operation_state<T>::value;

#define ASIO_EXECUTION_OPERATION_STATE \
::asio::execution::operation_state

#else 

#define ASIO_EXECUTION_OPERATION_STATE typename

#endif 

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 
