
#ifndef ASIO_POSIX_DESCRIPTOR_HPP
#define ASIO_POSIX_DESCRIPTOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_POSIX_STREAM_DESCRIPTOR) \
|| defined(GENERATING_DOCUMENTATION)

#include "asio/posix/basic_descriptor.hpp"

namespace asio {
namespace posix {

typedef basic_descriptor<> descriptor;

} 
} 

#endif 

#endif 
