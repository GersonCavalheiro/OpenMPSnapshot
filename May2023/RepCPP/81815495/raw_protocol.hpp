
#ifndef ASIO_GENERIC_RAW_PROTOCOL_HPP
#define ASIO_GENERIC_RAW_PROTOCOL_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#include <typeinfo>
#include "asio/basic_raw_socket.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/detail/throw_exception.hpp"
#include "asio/generic/basic_endpoint.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace generic {


class raw_protocol
{
public:
raw_protocol(int address_family, int socket_protocol)
: family_(address_family),
protocol_(socket_protocol)
{
}


template <typename Protocol>
raw_protocol(const Protocol& source_protocol)
: family_(source_protocol.family()),
protocol_(source_protocol.protocol())
{
if (source_protocol.type() != type())
{
std::bad_cast ex;
asio::detail::throw_exception(ex);
}
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

friend bool operator==(const raw_protocol& p1, const raw_protocol& p2)
{
return p1.family_ == p2.family_ && p1.protocol_ == p2.protocol_;
}

friend bool operator!=(const raw_protocol& p1, const raw_protocol& p2)
{
return !(p1 == p2);
}

typedef basic_endpoint<raw_protocol> endpoint;

typedef basic_raw_socket<raw_protocol> socket;

private:
int family_;
int protocol_;
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 
