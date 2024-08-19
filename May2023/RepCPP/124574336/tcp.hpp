
#ifndef BOOST_ASIO_IP_TCP_HPP
#define BOOST_ASIO_IP_TCP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/basic_socket_acceptor.hpp>
#include <boost/asio/basic_socket_iostream.hpp>
#include <boost/asio/basic_stream_socket.hpp>
#include <boost/asio/detail/socket_option.hpp>
#include <boost/asio/detail/socket_types.hpp>
#include <boost/asio/ip/basic_endpoint.hpp>
#include <boost/asio/ip/basic_resolver.hpp>
#include <boost/asio/ip/basic_resolver_iterator.hpp>
#include <boost/asio/ip/basic_resolver_query.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace ip {


class tcp
{
public:
typedef basic_endpoint<tcp> endpoint;

static tcp v4() BOOST_ASIO_NOEXCEPT
{
return tcp(BOOST_ASIO_OS_DEF(AF_INET));
}

static tcp v6() BOOST_ASIO_NOEXCEPT
{
return tcp(BOOST_ASIO_OS_DEF(AF_INET6));
}

int type() const BOOST_ASIO_NOEXCEPT
{
return BOOST_ASIO_OS_DEF(SOCK_STREAM);
}

int protocol() const BOOST_ASIO_NOEXCEPT
{
return BOOST_ASIO_OS_DEF(IPPROTO_TCP);
}

int family() const BOOST_ASIO_NOEXCEPT
{
return family_;
}

typedef basic_stream_socket<tcp> socket;

typedef basic_socket_acceptor<tcp> acceptor;

typedef basic_resolver<tcp> resolver;

#if !defined(BOOST_ASIO_NO_IOSTREAM)
typedef basic_socket_iostream<tcp> iostream;
#endif 


#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined no_delay;
#else
typedef boost::asio::detail::socket_option::boolean<
BOOST_ASIO_OS_DEF(IPPROTO_TCP), BOOST_ASIO_OS_DEF(TCP_NODELAY)> no_delay;
#endif

friend bool operator==(const tcp& p1, const tcp& p2)
{
return p1.family_ == p2.family_;
}

friend bool operator!=(const tcp& p1, const tcp& p2)
{
return p1.family_ != p2.family_;
}

private:
explicit tcp(int protocol_family) BOOST_ASIO_NOEXCEPT
: family_(protocol_family)
{
}

int family_;
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
