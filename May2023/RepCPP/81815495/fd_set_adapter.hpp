
#ifndef ASIO_DETAIL_FD_SET_ADAPTER_HPP
#define ASIO_DETAIL_FD_SET_ADAPTER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if !defined(ASIO_WINDOWS_RUNTIME)

#include "asio/detail/posix_fd_set_adapter.hpp"
#include "asio/detail/win_fd_set_adapter.hpp"

namespace asio {
namespace detail {

#if defined(ASIO_WINDOWS) || defined(__CYGWIN__)
typedef win_fd_set_adapter fd_set_adapter;
#else
typedef posix_fd_set_adapter fd_set_adapter;
#endif

} 
} 

#endif 

#endif 
