
#ifndef ASIO_DETAIL_POSIX_MUTEX_HPP
#define ASIO_DETAIL_POSIX_MUTEX_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_PTHREADS)

#include <pthread.h>
#include "asio/detail/noncopyable.hpp"
#include "asio/detail/scoped_lock.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class posix_event;

class posix_mutex
: private noncopyable
{
public:
typedef asio::detail::scoped_lock<posix_mutex> scoped_lock;

ASIO_DECL posix_mutex();

~posix_mutex()
{
::pthread_mutex_destroy(&mutex_); 
}

void lock()
{
(void)::pthread_mutex_lock(&mutex_); 
}

void unlock()
{
(void)::pthread_mutex_unlock(&mutex_); 
}

private:
friend class posix_event;
::pthread_mutex_t mutex_;
};

} 
} 

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/detail/impl/posix_mutex.ipp"
#endif 

#endif 

#endif 
