
#ifndef BOOST_ASIO_POSIX_DESCRIPTOR_BASE_HPP
#define BOOST_ASIO_POSIX_DESCRIPTOR_BASE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if defined(BOOST_ASIO_HAS_POSIX_STREAM_DESCRIPTOR) \
|| defined(GENERATING_DOCUMENTATION)

#include <boost/asio/detail/io_control.hpp>
#include <boost/asio/detail/socket_option.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
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
typedef boost::asio::detail::io_control::bytes_readable bytes_readable;
#endif

protected:
~descriptor_base()
{
}
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 

#endif 
