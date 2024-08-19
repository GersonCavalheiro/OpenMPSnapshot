
#ifndef BOOST_ASIO_DETAIL_NULL_THREAD_HPP
#define BOOST_ASIO_DETAIL_NULL_THREAD_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if !defined(BOOST_ASIO_HAS_THREADS)

#include <boost/asio/detail/noncopyable.hpp>
#include <boost/asio/detail/throw_error.hpp>
#include <boost/asio/error.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

class null_thread
: private noncopyable
{
public:
template <typename Function>
null_thread(Function, unsigned int = 0)
{
boost::asio::detail::throw_error(
boost::asio::error::operation_not_supported, "thread");
}

~null_thread()
{
}

void join()
{
}

static std::size_t hardware_concurrency()
{
return 1;
}
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 

#endif 
