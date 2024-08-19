
#ifndef ASIO_EXECUTION_SCHEDULER_HPP
#define ASIO_EXECUTION_SCHEDULER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/execution/schedule.hpp"
#include "asio/traits/equality_comparable.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace execution {
namespace detail {

template <typename T>
struct is_scheduler_base :
integral_constant<bool,
is_copy_constructible<typename remove_cvref<T>::type>::value
&& traits::equality_comparable<typename remove_cvref<T>::type>::is_valid
>
{
};

} 


template <typename T>
struct is_scheduler :
#if defined(GENERATING_DOCUMENTATION)
integral_constant<bool, automatically_determined>
#else 
conditional<
can_schedule<T>::value,
detail::is_scheduler_base<T>,
false_type
>::type
#endif 
{
};

#if defined(ASIO_HAS_VARIABLE_TEMPLATES)

template <typename T>
ASIO_CONSTEXPR const bool is_scheduler_v = is_scheduler<T>::value;

#endif 

#if defined(ASIO_HAS_CONCEPTS)

template <typename T>
ASIO_CONCEPT scheduler = is_scheduler<T>::value;

#define ASIO_EXECUTION_SCHEDULER ::asio::execution::scheduler

#else 

#define ASIO_EXECUTION_SCHEDULER typename

#endif 

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 
