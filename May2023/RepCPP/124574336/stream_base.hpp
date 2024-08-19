
#ifndef BOOST_ASIO_SSL_STREAM_BASE_HPP
#define BOOST_ASIO_SSL_STREAM_BASE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace ssl {

class stream_base
{
public:
enum handshake_type
{
client,

server
};

protected:
~stream_base()
{
}
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
