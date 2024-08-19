
#ifndef ASIO_BASIC_STREAMBUF_FWD_HPP
#define ASIO_BASIC_STREAMBUF_FWD_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if !defined(ASIO_NO_IOSTREAM)

#include <memory>

namespace asio {

template <typename Allocator = std::allocator<char> >
class basic_streambuf;

template <typename Allocator = std::allocator<char> >
class basic_streambuf_ref;

} 

#endif 

#endif 
