
#ifndef BOOST_ASIO_LOCAL_DATAGRAM_PROTOCOL_HPP
#define BOOST_ASIO_LOCAL_DATAGRAM_PROTOCOL_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if defined(BOOST_ASIO_HAS_LOCAL_SOCKETS) \
|| defined(GENERATING_DOCUMENTATION)

#include <boost/asio/basic_datagram_socket.hpp>
#include <boost/asio/detail/socket_types.hpp>
#include <boost/asio/local/basic_endpoint.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace local {


class datagram_protocol
{
public:
int type() const BOOST_ASIO_NOEXCEPT
{
return SOCK_DGRAM;
}

int protocol() const BOOST_ASIO_NOEXCEPT
{
return 0;
}

int family() const BOOST_ASIO_NOEXCEPT
{
return AF_UNIX;
}

typedef basic_endpoint<datagram_protocol> endpoint;

typedef basic_datagram_socket<datagram_protocol> socket;
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 

#endif 
