
#ifndef BOOST_ASIO_DETAIL_FD_SET_ADAPTER_HPP
#define BOOST_ASIO_DETAIL_FD_SET_ADAPTER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if !defined(BOOST_ASIO_WINDOWS_RUNTIME)

#include <boost/asio/detail/posix_fd_set_adapter.hpp>
#include <boost/asio/detail/win_fd_set_adapter.hpp>

namespace boost {
namespace asio {
namespace detail {

#if defined(BOOST_ASIO_WINDOWS) || defined(__CYGWIN__)
typedef win_fd_set_adapter fd_set_adapter;
#else
typedef posix_fd_set_adapter fd_set_adapter;
#endif

} 
} 
} 

#endif 

#endif 
