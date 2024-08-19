
#ifndef ASIO_IP_ICMP_HPP
#define ASIO_IP_ICMP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/basic_raw_socket.hpp"
#include "asio/ip/basic_endpoint.hpp"
#include "asio/ip/basic_resolver.hpp"
#include "asio/ip/basic_resolver_iterator.hpp"
#include "asio/ip/basic_resolver_query.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ip {


class icmp
{
public:
typedef basic_endpoint<icmp> endpoint;

static icmp v4() ASIO_NOEXCEPT
{
return icmp(ASIO_OS_DEF(IPPROTO_ICMP),
ASIO_OS_DEF(AF_INET));
}

static icmp v6() ASIO_NOEXCEPT
{
return icmp(ASIO_OS_DEF(IPPROTO_ICMPV6),
ASIO_OS_DEF(AF_INET6));
}

int type() const ASIO_NOEXCEPT
{
return ASIO_OS_DEF(SOCK_RAW);
}

int protocol() const ASIO_NOEXCEPT
{
return protocol_;
}

int family() const ASIO_NOEXCEPT
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
explicit icmp(int protocol_id, int protocol_family) ASIO_NOEXCEPT
: protocol_(protocol_id),
family_(protocol_family)
{
}

int protocol_;
int family_;
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 
