
#ifndef BOOST_ASIO_IP_ICMP_HPP
#define BOOST_ASIO_IP_ICMP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/detail/socket_types.hpp>
#include <boost/asio/basic_raw_socket.hpp>
#include <boost/asio/ip/basic_endpoint.hpp>
#include <boost/asio/ip/basic_resolver.hpp>
#include <boost/asio/ip/basic_resolver_iterator.hpp>
#include <boost/asio/ip/basic_resolver_query.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace ip {


class icmp
{
public:
typedef basic_endpoint<icmp> endpoint;

static icmp v4() BOOST_ASIO_NOEXCEPT
{
return icmp(BOOST_ASIO_OS_DEF(IPPROTO_ICMP),
BOOST_ASIO_OS_DEF(AF_INET));
}

static icmp v6() BOOST_ASIO_NOEXCEPT
{
return icmp(BOOST_ASIO_OS_DEF(IPPROTO_ICMPV6),
BOOST_ASIO_OS_DEF(AF_INET6));
}

int type() const BOOST_ASIO_NOEXCEPT
{
return BOOST_ASIO_OS_DEF(SOCK_RAW);
}

int protocol() const BOOST_ASIO_NOEXCEPT
{
return protocol_;
}

int family() const BOOST_ASIO_NOEXCEPT
{
return family_;
}

typedef basic_raw_socket<icmp> socket;

typedef basic_resolver<icmp> resolver;

friend bool operator==(const icmp& p1, const icmp& p2)
{
return p1.protocol_ == p2.protocol_ && p1.family_ == p2.family_;
}

friend bool operator!=(const icmp& p1, const icmp& p2)
{
return p1.protocol_ != p2.protocol_ || p1.family_ != p2.family_;
}

private:
explicit icmp(int protocol_id, int protocol_family) BOOST_ASIO_NOEXCEPT
: protocol_(protocol_id),
family_(protocol_family)
{
}

int protocol_;
int family_;
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
