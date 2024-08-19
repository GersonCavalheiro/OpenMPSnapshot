
#ifndef ASIO_SSL_STREAM_BASE_HPP
#define ASIO_SSL_STREAM_BASE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#include "asio/detail/push_options.hpp"

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

#include "asio/detail/pop_options.hpp"

#endif 
