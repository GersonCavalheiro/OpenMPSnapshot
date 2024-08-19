
#ifndef ASIO_DETAIL_NULL_MUTEX_HPP
#define ASIO_DETAIL_NULL_MUTEX_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if !defined(ASIO_HAS_THREADS)

#include "asio/detail/noncopyable.hpp"
#include "asio/detail/scoped_lock.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class null_mutex
: private noncopyable
{
public:
typedef asio::detail::scoped_lock<null_mutex> scoped_lock;

null_mutex()
{
}

~null_mutex()
{
}

void lock()
{
}

void unlock()
{
}
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 

#endif 
