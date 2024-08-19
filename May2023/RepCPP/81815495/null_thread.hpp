
#ifndef ASIO_DETAIL_NULL_THREAD_HPP
#define ASIO_DETAIL_NULL_THREAD_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if !defined(ASIO_HAS_THREADS)

#include "asio/detail/noncopyable.hpp"
#include "asio/detail/throw_error.hpp"
#include "asio/error.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class null_thread
: private noncopyable
{
public:
template <typename Function>
null_thread(Function, unsigned int = 0)
{
asio::detail::throw_error(
asio::error::operation_not_supported, "thread");
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

#include "asio/detail/pop_options.hpp"

#endif 

#endif 
