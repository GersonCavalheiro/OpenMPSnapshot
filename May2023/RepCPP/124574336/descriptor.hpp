
#ifndef BOOST_ASIO_POSIX_DESCRIPTOR_HPP
#define BOOST_ASIO_POSIX_DESCRIPTOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if defined(BOOST_ASIO_HAS_POSIX_STREAM_DESCRIPTOR) \
|| defined(GENERATING_DOCUMENTATION)

#include <boost/asio/posix/basic_descriptor.hpp>

namespace boost {
namespace asio {
namespace posix {

typedef basic_descriptor<> descriptor;

} 
} 
} 

#endif 

#endif 
