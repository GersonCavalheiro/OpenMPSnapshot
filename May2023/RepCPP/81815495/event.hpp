
#ifndef ASIO_DETAIL_EVENT_HPP
#define ASIO_DETAIL_EVENT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if !defined(ASIO_HAS_THREADS)
# include "asio/detail/null_event.hpp"
#elif defined(ASIO_WINDOWS)
# include "asio/detail/win_event.hpp"
#elif defined(ASIO_HAS_PTHREADS)
# include "asio/detail/posix_event.hpp"
#elif defined(ASIO_HAS_STD_MUTEX_AND_CONDVAR)
# include "asio/detail/std_event.hpp"
#else
# error Only Windows, POSIX and std::condition_variable are supported!
#endif

namespace asio {
namespace detail {

#if !defined(ASIO_HAS_THREADS)
typedef null_event event;
#elif defined(ASIO_WINDOWS)
typedef win_event event;
#elif defined(ASIO_HAS_PTHREADS)
typedef posix_event event;
#elif defined(ASIO_HAS_STD_MUTEX_AND_CONDVAR)
typedef std_event event;
#endif

} 
} 

#endif 
