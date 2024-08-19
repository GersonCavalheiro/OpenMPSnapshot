
#ifndef ASIO_POSIX_DESCRIPTOR_BASE_HPP
#define ASIO_POSIX_DESCRIPTOR_BASE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_POSIX_STREAM_DESCRIPTOR) \
|| defined(GENERATING_DOCUMENTATION)

#include "asio/detail/io_control.hpp"
#include "asio/detail/socket_option.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace posix {

class descriptor_base
{
public:

enum wait_type
{
wait_read,

wait_write,

wait_error
};


#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined bytes_readable;
#else
typedef asio::detail::io_control::bytes_readable bytes_readable;
#endif

protected:
~descriptor_base()
{
}
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 

#endif 
