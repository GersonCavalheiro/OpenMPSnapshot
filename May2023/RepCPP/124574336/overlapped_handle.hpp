
#ifndef BOOST_ASIO_WINDOWS_OVERLAPPED_HANDLE_HPP
#define BOOST_ASIO_WINDOWS_OVERLAPPED_HANDLE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if defined(BOOST_ASIO_HAS_WINDOWS_RANDOM_ACCESS_HANDLE) \
|| defined(BOOST_ASIO_HAS_WINDOWS_STREAM_HANDLE) \
|| defined(GENERATING_DOCUMENTATION)

#include <boost/asio/windows/basic_overlapped_handle.hpp>

namespace boost {
namespace asio {
namespace windows {

typedef basic_overlapped_handle<> overlapped_handle;

} 
} 
} 

#endif 

#endif 
