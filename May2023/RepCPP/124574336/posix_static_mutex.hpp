
#ifndef BOOST_ASIO_DETAIL_POSIX_STATIC_MUTEX_HPP
#define BOOST_ASIO_DETAIL_POSIX_STATIC_MUTEX_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if defined(BOOST_ASIO_HAS_PTHREADS)

#include <pthread.h>
#include <boost/asio/detail/scoped_lock.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

struct posix_static_mutex
{
typedef boost::asio::detail::scoped_lock<posix_static_mutex> scoped_lock;

void init()
{
}

void lock()
{
(void)::pthread_mutex_lock(&mutex_); 
}

void unlock()
{
(void)::pthread_mutex_unlock(&mutex_); 
}

::pthread_mutex_t mutex_;
};

#define BOOST_ASIO_POSIX_STATIC_MUTEX_INIT { PTHREAD_MUTEX_INITIALIZER }

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 

#endif 
