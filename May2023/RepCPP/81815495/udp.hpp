
#ifndef ASIO_IP_UDP_HPP
#define ASIO_IP_UDP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/basic_datagram_socket.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/ip/basic_endpoint.hpp"
#include "asio/ip/basic_resolver.hpp"
#include "asio/ip/basic_resolver_iterator.hpp"
#include "asio/ip/basic_resolver_query.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ip {


class udp
{
public:
typedef basic_endpoint<udp> endpoint;

static udp v4() ASIO_NOEXCEPT
{
return udp(ASIO_OS_DEF(AF_INET));
}

static udp v6() ASIO_NOEXCEPT
{
return udp(ASIO_OS_DEF(AF_INET6));
}

int type() const ASIO_NOEXCEPT
{
return ASIO_OS_DEF(SOCK_DGRAM);
}

int protocol() const ASIO_NOEXCEPT
{
return ASIO_OS_DEF(IPPROTO_UDP);
}

int family() const ASIO_NOEXCEPT
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
explicit udp(int protocol_family) ASIO_NOEXCEPT
: family_(protocol_family)
{
}

int family_;
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 
