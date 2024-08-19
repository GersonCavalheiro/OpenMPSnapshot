
#ifndef BOOST_ASIO_IP_BASIC_RESOLVER_ENTRY_HPP
#define BOOST_ASIO_IP_BASIC_RESOLVER_ENTRY_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <string>
#include <boost/asio/detail/string_view.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace ip {


template <typename InternetProtocol>
class basic_resolver_entry
{
public:
typedef InternetProtocol protocol_type;

typedef typename InternetProtocol::endpoint endpoint_type;

basic_resolver_entry()
{
}

basic_resolver_entry(const endpoint_type& ep,
BOOST_ASIO_STRING_VIEW_PARAM host, BOOST_ASIO_STRING_VIEW_PARAM service)
: endpoint_(ep),
host_name_(static_cast<std::string>(host)),
service_name_(static_cast<std::string>(service))
{
}

endpoint_type endpoint() const
{
return endpoint_;
}

operator endpoint_type() const
{
return endpoint_;
}

std::string host_name() const
{
return host_name_;
}

template <class Allocator>
std::basic_string<char, std::char_traits<char>, Allocator> host_name(
const Allocator& alloc = Allocator()) const
{
return std::basic_string<char, std::char_traits<char>, Allocator>(
host_name_.c_str(), alloc);
}

std::string service_name() const
{
return service_name_;
}

template <class Allocator>
std::basic_string<char, std::char_traits<char>, Allocator> service_name(
const Allocator& alloc = Allocator()) const
{
return std::basic_string<char, std::char_traits<char>, Allocator>(
service_name_.c_str(), alloc);
}

private:
endpoint_type endpoint_;
std::string host_name_;
std::string service_name_;
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
