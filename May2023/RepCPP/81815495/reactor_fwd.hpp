
#ifndef ASIO_DETAIL_REACTOR_FWD_HPP
#define ASIO_DETAIL_REACTOR_FWD_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

namespace asio {
namespace detail {

#if defined(ASIO_HAS_IOCP) || defined(ASIO_WINDOWS_RUNTIME)
typedef class null_reactor reactor;
#elif defined(ASIO_HAS_IOCP)
typedef class select_reactor reactor;
#elif defined(ASIO_HAS_EPOLL)
typedef class epoll_reactor reactor;
#elif defined(ASIO_HAS_KQUEUE)
typedef class kqueue_reactor reactor;
#elif defined(ASIO_HAS_DEV_POLL)
typedef class dev_poll_reactor reactor;
#else
typedef class select_reactor reactor;
#endif

} 
} 

#endif 
