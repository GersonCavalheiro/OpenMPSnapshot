
#ifndef ASIO_IP_TCP_HPP
#define ASIO_IP_TCP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/basic_socket_acceptor.hpp"
#include "asio/basic_socket_iostream.hpp"
#include "asio/basic_stream_socket.hpp"
#include "asio/detail/socket_option.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/ip/basic_endpoint.hpp"
#include "asio/ip/basic_resolver.hpp"
#include "asio/ip/basic_resolver_iterator.hpp"
#include "asio/ip/basic_resolver_query.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ip {


class tcp
{
public:
typedef basic_endpoint<tcp> endpoint;

static tcp v4() ASIO_NOEXCEPT
{
return tcp(ASIO_OS_DEF(AF_INET));
}

static tcp v6() ASIO_NOEXCEPT
{
return tcp(ASIO_OS_DEF(AF_INET6));
}

int type() const ASIO_NOEXCEPT
{
return ASIO_OS_DEF(SOCK_STREAM);
}

int protocol() const ASIO_NOEXCEPT
{
return ASIO_OS_DEF(IPPROTO_TCP);
}

int family() const ASIO_NOEXCEPT
{
return family_;
}

typedef basic_stream_socket<tcp> socket;

typedef basic_socket_acceptor<tcp> acceptor;

typedef basic_resolver<tcp> resolver;

#if !defined(ASIO_NO_IOSTREAM)
typedef basic_socket_iostream<tcp> iostream;
#endif 


#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined no_delay;
#else
typedef asio::detail::socket_option::boolean<
ASIO_OS_DEF(IPPROTO_TCP), ASIO_OS_DEF(TCP_NODELAY)> no_delay;
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
explicit tcp(int protocol_family) ASIO_NOEXCEPT
: family_(protocol_family)
{
}

int family_;
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 
