
#ifndef ASIO_DETAIL_TIMER_SCHEDULER_HPP
#define ASIO_DETAIL_TIMER_SCHEDULER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/detail/timer_scheduler_fwd.hpp"

#if defined(ASIO_WINDOWS_RUNTIME)
# include "asio/detail/winrt_timer_scheduler.hpp"
#elif defined(ASIO_HAS_IOCP)
# include "asio/detail/win_iocp_io_context.hpp"
#elif defined(ASIO_HAS_EPOLL)
# include "asio/detail/epoll_reactor.hpp"
#elif defined(ASIO_HAS_KQUEUE)
# include "asio/detail/kqueue_reactor.hpp"
#elif defined(ASIO_HAS_DEV_POLL)
# include "asio/detail/dev_poll_reactor.hpp"
#else
# include "asio/detail/select_reactor.hpp"
#endif

#endif 
