
#ifndef BOOST_ASIO_USES_EXECUTOR_HPP
#define BOOST_ASIO_USES_EXECUTOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/detail/type_traits.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {


struct executor_arg_t
{
BOOST_ASIO_CONSTEXPR executor_arg_t() BOOST_ASIO_NOEXCEPT
{
}
};


#if defined(BOOST_ASIO_HAS_CONSTEXPR) || defined(GENERATING_DOCUMENTATION)
constexpr executor_arg_t executor_arg;
#elif defined(BOOST_ASIO_MSVC)
__declspec(selectany) executor_arg_t executor_arg;
#endif


template <typename T, typename Executor>
struct uses_executor : false_type {};

} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
