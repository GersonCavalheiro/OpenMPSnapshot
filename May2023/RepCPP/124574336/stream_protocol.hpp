
#ifndef BOOST_ASIO_LOCAL_STREAM_PROTOCOL_HPP
#define BOOST_ASIO_LOCAL_STREAM_PROTOCOL_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if defined(BOOST_ASIO_HAS_LOCAL_SOCKETS) \
|| defined(GENERATING_DOCUMENTATION)

#include <boost/asio/basic_socket_acceptor.hpp>
#include <boost/asio/basic_socket_iostream.hpp>
#include <boost/asio/basic_stream_socket.hpp>
#include <boost/asio/detail/socket_types.hpp>
#include <boost/asio/local/basic_endpoint.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace local {


class stream_protocol
{
public:
int type() const BOOST_ASIO_NOEXCEPT
{
return SOCK_STREAM;
}

int protocol() const BOOST_ASIO_NOEXCEPT
{
return 0;
}

int family() const BOOST_ASIO_NOEXCEPT
{
return AF_UNIX;
}

typedef basic_endpoint<stream_protocol> endpoint;

typedef basic_stream_socket<stream_protocol> socket;

typedef basic_socket_acceptor<stream_protocol> acceptor;

#if !defined(BOOST_ASIO_NO_IOSTREAM)
typedef basic_socket_iostream<stream_protocol> iostream;
#endif 
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 

#endif 
