
#ifndef ASIO_WINDOWS_STREAM_HANDLE_HPP
#define ASIO_WINDOWS_STREAM_HANDLE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_WINDOWS_STREAM_HANDLE) \
|| defined(GENERATING_DOCUMENTATION)

#include "asio/windows/basic_stream_handle.hpp"

namespace asio {
namespace windows {

typedef basic_stream_handle<> stream_handle;

} 
} 

#endif 

#endif 
