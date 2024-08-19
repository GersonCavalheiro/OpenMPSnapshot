
#ifndef BOOST_ASIO_IP_UDP_HPP
#define BOOST_ASIO_IP_UDP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/basic_datagram_socket.hpp>
#include <boost/asio/detail/socket_types.hpp>
#include <boost/asio/ip/basic_endpoint.hpp>
#include <boost/asio/ip/basic_resolver.hpp>
#include <boost/asio/ip/basic_resolver_iterator.hpp>
#include <boost/asio/ip/basic_resolver_query.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace ip {


class udp
{
public:
typedef basic_endpoint<udp> endpoint;

static udp v4() BOOST_ASIO_NOEXCEPT
{
return udp(BOOST_ASIO_OS_DEF(AF_INET));
}

static udp v6() BOOST_ASIO_NOEXCEPT
{
return udp(BOOST_ASIO_OS_DEF(AF_INET6));
}

int type() const BOOST_ASIO_NOEXCEPT
{
return BOOST_ASIO_OS_DEF(SOCK_DGRAM);
}

int protocol() const BOOST_ASIO_NOEXCEPT
{
return BOOST_ASIO_OS_DEF(IPPROTO_UDP);
}

int family() const BOOST_ASIO_NOEXCEPT
{
return family_;
}

typedef basic_datagram_socket<udp> socket;

typedef basic_resolver<udp> resolver;

friend bool operator==(const udp& p1, const udp& p2)
{
return p1.family_ == p2.family_;
}

friend bool operator!=(const udp& p1, const udp& p2)
{
return p1.family_ != p2.family_;
}

private:
explicit udp(int protocol_family) BOOST_ASIO_NOEXCEPT
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
