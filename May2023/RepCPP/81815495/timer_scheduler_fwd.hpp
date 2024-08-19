
#ifndef ASIO_DETAIL_TIMER_SCHEDULER_FWD_HPP
#define ASIO_DETAIL_TIMER_SCHEDULER_FWD_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

namespace asio {
namespace detail {

#if defined(ASIO_WINDOWS_RUNTIME)
typedef class winrt_timer_scheduler timer_scheduler;
#elif defined(ASIO_HAS_IOCP)
typedef class win_iocp_io_context timer_scheduler;
#elif defined(ASIO_HAS_EPOLL)
typedef class epoll_reactor timer_scheduler;
#elif defined(ASIO_HAS_KQUEUE)
typedef class kqueue_reactor timer_scheduler;
#elif defined(ASIO_HAS_DEV_POLL)
typedef class dev_poll_reactor timer_scheduler;
#else
typedef class select_reactor timer_scheduler;
#endif

} 
} 

#endif 
