
#ifndef ASIO_LOCAL_DATAGRAM_PROTOCOL_HPP
#define ASIO_LOCAL_DATAGRAM_PROTOCOL_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_LOCAL_SOCKETS) \
|| defined(GENERATING_DOCUMENTATION)

#include "asio/basic_datagram_socket.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/local/basic_endpoint.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace local {


class datagram_protocol
{
public:
int type() const ASIO_NOEXCEPT
{
return SOCK_DGRAM;
}

int protocol() const ASIO_NOEXCEPT
{
return 0;
}

int family() const ASIO_NOEXCEPT
{
return AF_UNIX;
}

typedef basic_endpoint<datagram_protocol> endpoint;

typedef basic_datagram_socket<datagram_protocol> socket;
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 

#endif 
