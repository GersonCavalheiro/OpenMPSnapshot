
#ifndef ASIO_DETAIL_SELECT_INTERRUPTER_HPP
#define ASIO_DETAIL_SELECT_INTERRUPTER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if !defined(ASIO_WINDOWS_RUNTIME)

#if defined(ASIO_WINDOWS) || defined(__CYGWIN__) || defined(__SYMBIAN32__)
# include "asio/detail/socket_select_interrupter.hpp"
#elif defined(ASIO_HAS_EVENTFD)
# include "asio/detail/eventfd_select_interrupter.hpp"
#else
# include "asio/detail/pipe_select_interrupter.hpp"
#endif

namespace asio {
namespace detail {

#if defined(ASIO_WINDOWS) || defined(__CYGWIN__) || defined(__SYMBIAN32__)
typedef socket_select_interrupter select_interrupter;
#elif defined(ASIO_HAS_EVENTFD)
typedef eventfd_select_interrupter select_interrupter;
#else
typedef pipe_select_interrupter select_interrupter;
#endif

} 
} 

#endif 

#endif 
