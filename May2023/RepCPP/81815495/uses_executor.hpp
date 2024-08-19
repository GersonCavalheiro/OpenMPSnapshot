
#ifndef ASIO_USES_EXECUTOR_HPP
#define ASIO_USES_EXECUTOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/detail/type_traits.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {


struct executor_arg_t
{
ASIO_CONSTEXPR executor_arg_t() ASIO_NOEXCEPT
{
}
};


#if defined(ASIO_HAS_CONSTEXPR) || defined(GENERATING_DOCUMENTATION)
constexpr executor_arg_t executor_arg;
#elif defined(ASIO_MSVC)
__declspec(selectany) executor_arg_t executor_arg;
#endif


template <typename T, typename Executor>
struct uses_executor : false_type {};

} 

#include "asio/detail/pop_options.hpp"

#endif 
