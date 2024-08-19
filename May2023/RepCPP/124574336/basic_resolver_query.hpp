
#ifndef BOOST_ASIO_IP_BASIC_RESOLVER_QUERY_HPP
#define BOOST_ASIO_IP_BASIC_RESOLVER_QUERY_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <string>
#include <boost/asio/detail/socket_ops.hpp>
#include <boost/asio/ip/resolver_query_base.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace ip {


template <typename InternetProtocol>
class basic_resolver_query
: public resolver_query_base
{
public:
typedef InternetProtocol protocol_type;


basic_resolver_query(const std::string& service,
resolver_query_base::flags resolve_flags = passive | address_configured)
: hints_(),
host_name_(),
service_name_(service)
{
typename InternetProtocol::endpoint endpoint;
hints_.ai_flags = static_cast<int>(resolve_flags);
hints_.ai_family = PF_UNSPEC;
hints_.ai_socktype = endpoint.protocol().type();
hints_.ai_protocol = endpoint.protocol().protocol();
hints_.ai_addrlen = 0;
hints_.ai_canonname = 0;
hints_.ai_addr = 0;
hints_.ai_next = 0;
}


basic_resolver_query(const protocol_type& protocol,
const std::string& service,
resolver_query_base::flags resolve_flags = passive | address_configured)
: hints_(),
host_name_(),
service_name_(service)
{
hints_.ai_flags = static_cast<int>(resolve_flags);
hints_.ai_family = protocol.family();
hints_.ai_socktype = protocol.type();
hints_.ai_protocol = protocol.protocol();
hints_.ai_addrlen = 0;
hints_.ai_canonname = 0;
hints_.ai_addr = 0;
hints_.ai_next = 0;
}


basic_resolver_query(const std::string& host, const std::string& service,
resolver_query_base::flags resolve_flags = address_configured)
: hints_(),
host_name_(host),
service_name_(service)
{
typename InternetProtocol::endpoint endpoint;
hints_.ai_flags = static_cast<int>(resolve_flags);
hints_.ai_family = BOOST_ASIO_OS_DEF(AF_UNSPEC);
hints_.ai_socktype = endpoint.protocol().type();
hints_.ai_protocol = endpoint.protocol().protocol();
hints_.ai_addrlen = 0;
hints_.ai_canonname = 0;
hints_.ai_addr = 0;
hints_.ai_next = 0;
}


basic_resolver_query(const protocol_type& protocol,
const std::string& host, const std::string& service,
resolver_query_base::flags resolve_flags = address_configured)
: hints_(),
host_name_(host),
service_name_(service)
{
hints_.ai_flags = static_cast<int>(resolve_flags);
hints_.ai_family = protocol.family();
hints_.ai_socktype = protocol.type();
hints_.ai_protocol = protocol.protocol();
hints_.ai_addrlen = 0;
hints_.ai_canonname = 0;
hints_.ai_addr = 0;
hints_.ai_next = 0;
}

const boost::asio::detail::addrinfo_type& hints() const
{
return hints_;
}

std::string host_name() const
{
return host_name_;
}

std::string service_name() const
{
return service_name_;
}

private:
boost::asio::detail::addrinfo_type hints_;
std::string host_name_;
std::string service_name_;
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
