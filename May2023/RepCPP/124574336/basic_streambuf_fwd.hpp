
#ifndef BOOST_ASIO_BASIC_STREAMBUF_FWD_HPP
#define BOOST_ASIO_BASIC_STREAMBUF_FWD_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if !defined(BOOST_ASIO_NO_IOSTREAM)

#include <memory>

namespace boost {
namespace asio {

template <typename Allocator = std::allocator<char> >
class basic_streambuf;

template <typename Allocator = std::allocator<char> >
class basic_streambuf_ref;

} 
} 

#endif 

#endif 
