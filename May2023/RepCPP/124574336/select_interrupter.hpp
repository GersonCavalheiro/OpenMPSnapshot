
#ifndef BOOST_ASIO_DETAIL_SELECT_INTERRUPTER_HPP
#define BOOST_ASIO_DETAIL_SELECT_INTERRUPTER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if !defined(BOOST_ASIO_WINDOWS_RUNTIME)

#if defined(BOOST_ASIO_WINDOWS) || defined(__CYGWIN__) || defined(__SYMBIAN32__)
# include <boost/asio/detail/socket_select_interrupter.hpp>
#elif defined(BOOST_ASIO_HAS_EVENTFD)
# include <boost/asio/detail/eventfd_select_interrupter.hpp>
#else
# include <boost/asio/detail/pipe_select_interrupter.hpp>
#endif

namespace boost {
namespace asio {
namespace detail {

#if defined(BOOST_ASIO_WINDOWS) || defined(__CYGWIN__) || defined(__SYMBIAN32__)
typedef socket_select_interrupter select_interrupter;
#elif defined(BOOST_ASIO_HAS_EVENTFD)
typedef eventfd_select_interrupter select_interrupter;
#else
typedef pipe_select_interrupter select_interrupter;
#endif

} 
} 
} 

#endif 

#endif 
