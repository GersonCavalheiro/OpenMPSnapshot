
#ifndef ASIO_LOCAL_STREAM_PROTOCOL_HPP
#define ASIO_LOCAL_STREAM_PROTOCOL_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_LOCAL_SOCKETS) \
|| defined(GENERATING_DOCUMENTATION)

#include "asio/basic_socket_acceptor.hpp"
#include "asio/basic_socket_iostream.hpp"
#include "asio/basic_stream_socket.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/local/basic_endpoint.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace local {


class stream_protocol
{
public:
int type() const ASIO_NOEXCEPT
{
return SOCK_STREAM;
}

int protocol() const ASIO_NOEXCEPT
{
return 0;
}

int family() const ASIO_NOEXCEPT
{
return AF_UNIX;
}

typedef basic_endpoint<stream_protocol> endpoint;

typedef basic_stream_socket<stream_protocol> socket;

typedef basic_socket_acceptor<stream_protocol> acceptor;

#if !defined(ASIO_NO_IOSTREAM)
typedef basic_socket_iostream<stream_protocol> iostream;
#endif 
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 

#endif 
