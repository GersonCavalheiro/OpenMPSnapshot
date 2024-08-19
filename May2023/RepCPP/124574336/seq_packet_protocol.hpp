
#ifndef BOOST_ASIO_GENERIC_SEQ_PACKET_PROTOCOL_HPP
#define BOOST_ASIO_GENERIC_SEQ_PACKET_PROTOCOL_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#include <typeinfo>
#include <boost/asio/basic_seq_packet_socket.hpp>
#include <boost/asio/detail/socket_types.hpp>
#include <boost/asio/detail/throw_exception.hpp>
#include <boost/asio/generic/basic_endpoint.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace generic {


class seq_packet_protocol
{
public:
seq_packet_protocol(int address_family, int socket_protocol)
: family_(address_family),
protocol_(socket_protocol)
{
}


template <typename Protocol>
seq_packet_protocol(const Protocol& source_protocol)
: family_(source_protocol.family()),
protocol_(source_protocol.protocol())
{
if (source_protocol.type() != type())
{
std::bad_cast ex;
boost::asio::detail::throw_exception(ex);
}
}

int type() const BOOST_ASIO_NOEXCEPT
{
return BOOST_ASIO_OS_DEF(SOCK_SEQPACKET);
}

int protocol() const BOOST_ASIO_NOEXCEPT
{
return protocol_;
}

int family() const BOOST_ASIO_NOEXCEPT
{
return family_;
}

friend bool operator==(const seq_packet_protocol& p1,
const seq_packet_protocol& p2)
{
return p1.family_ == p2.family_ && p1.protocol_ == p2.protocol_;
}

friend bool operator!=(const seq_packet_protocol& p1,
const seq_packet_protocol& p2)
{
return !(p1 == p2);
}

typedef basic_endpoint<seq_packet_protocol> endpoint;

typedef basic_seq_packet_socket<seq_packet_protocol> socket;

private:
int family_;
int protocol_;
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
