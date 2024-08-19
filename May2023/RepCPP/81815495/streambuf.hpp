
#ifndef ASIO_STREAMBUF_HPP
#define ASIO_STREAMBUF_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if !defined(ASIO_NO_IOSTREAM)

#include "asio/basic_streambuf.hpp"

namespace asio {

typedef basic_streambuf<> streambuf;

} 

#endif 

#endif 
