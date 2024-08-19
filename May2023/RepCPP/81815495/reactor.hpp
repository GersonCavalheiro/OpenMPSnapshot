
#ifndef ASIO_DETAIL_REACTOR_HPP
#define ASIO_DETAIL_REACTOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/reactor_fwd.hpp"

#if defined(ASIO_HAS_EPOLL)
# include "asio/detail/epoll_reactor.hpp"
#elif defined(ASIO_HAS_KQUEUE)
# include "asio/detail/kqueue_reactor.hpp"
#elif defined(ASIO_HAS_DEV_POLL)
# include "asio/detail/dev_poll_reactor.hpp"
#elif defined(ASIO_HAS_IOCP) || defined(ASIO_WINDOWS_RUNTIME)
# include "asio/detail/null_reactor.hpp"
#else
# include "asio/detail/select_reactor.hpp"
#endif

#endif 
