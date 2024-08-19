
#ifndef BOOST_ASIO_IP_BASIC_RESOLVER_RESULTS_HPP
#define BOOST_ASIO_IP_BASIC_RESOLVER_RESULTS_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <cstddef>
#include <cstring>
#include <boost/asio/detail/socket_ops.hpp>
#include <boost/asio/detail/socket_types.hpp>
#include <boost/asio/ip/basic_resolver_iterator.hpp>

#if defined(BOOST_ASIO_WINDOWS_RUNTIME)
# include <boost/asio/detail/winrt_utils.hpp>
#endif 

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace ip {


template <typename InternetProtocol>
class basic_resolver_results
#if !defined(BOOST_ASIO_NO_DEPRECATED)
: public basic_resolver_iterator<InternetProtocol>
#else 
: private basic_resolver_iterator<InternetProtocol>
#endif 
{
public:
typedef InternetProtocol protocol_type;

typedef typename protocol_type::endpoint endpoint_type;

typedef basic_resolver_entry<protocol_type> value_type;

typedef const value_type& const_reference;

typedef value_type& reference;

typedef basic_resolver_iterator<protocol_type> const_iterator;

typedef const_iterator iterator;

typedef std::ptrdiff_t difference_type;

typedef std::size_t size_type;

basic_resolver_results()
{
}

basic_resolver_results(const basic_resolver_results& other)
: basic_resolver_iterator<InternetProtocol>(other)
{
}

#if defined(BOOST_ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)
basic_resolver_results(basic_resolver_results&& other)
: basic_resolver_iterator<InternetProtocol>(
BOOST_ASIO_MOVE_CAST(basic_resolver_results)(other))
{
}
#endif 

basic_resolver_results& operator=(const basic_resolver_results& other)
{
basic_resolver_iterator<InternetProtocol>::operator=(other);
return *this;
}

#if defined(BOOST_ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)
basic_resolver_results& operator=(basic_resolver_results&& other)
{
basic_resolver_iterator<InternetProtocol>::operator=(
BOOST_ASIO_MOVE_CAST(basic_resolver_results)(other));
return *this;
}
#endif 

#if !defined(GENERATING_DOCUMENTATION)
static basic_resolver_results create(
boost::asio::detail::addrinfo_type* address_info,
const std::string& host_name, const std::string& service_name)
{
basic_resolver_results results;
if (!address_info)
return results;

std::string actual_host_name = host_name;
if (address_info->ai_canonname)
actual_host_name = address_info->ai_canonname;

results.values_.reset(new values_type);

while (address_info)
{
if (address_info->ai_family == BOOST_ASIO_OS_DEF(AF_INET)
|| address_info->ai_family == BOOST_ASIO_OS_DEF(AF_INET6))
{
using namespace std; 
typename InternetProtocol::endpoint endpoint;
endpoint.resize(static_cast<std::size_t>(address_info->ai_addrlen));
memcpy(endpoint.data(), address_info->ai_addr,
address_info->ai_addrlen);
results.values_->push_back(
basic_resolver_entry<InternetProtocol>(endpoint,
actual_host_name, service_name));
}
address_info = address_info->ai_next;
}

return results;
}

static basic_resolver_results create(const endpoint_type& endpoint,
const std::string& host_name, const std::string& service_name)
{
basic_resolver_results results;
results.values_.reset(new values_type);
results.values_->push_back(
basic_resolver_entry<InternetProtocol>(
endpoint, host_name, service_name));
return results;
}

template <typename EndpointIterator>
static basic_resolver_results create(
EndpointIterator begin, EndpointIterator end,
const std::string& host_name, const std::string& service_name)
{
basic_resolver_results results;
if (begin != end)
{
results.values_.reset(new values_type);
for (EndpointIterator ep_iter = begin; ep_iter != end; ++ep_iter)
{
results.values_->push_back(
basic_resolver_entry<InternetProtocol>(
*ep_iter, host_name, service_name));
}
}
return results;
}

# if defined(BOOST_ASIO_WINDOWS_RUNTIME)
static basic_resolver_results create(
Windows::Foundation::Collections::IVectorView<
Windows::Networking::EndpointPair^>^ endpoints,
const boost::asio::detail::addrinfo_type& hints,
const std::string& host_name, const std::string& service_name)
{
basic_resolver_results results;
if (endpoints->Size)
{
results.values_.reset(new values_type);
for (unsigned int i = 0; i < endpoints->Size; ++i)
{
auto pair = endpoints->GetAt(i);

if (hints.ai_family == BOOST_ASIO_OS_DEF(AF_INET)
&& pair->RemoteHostName->Type
!= Windows::Networking::HostNameType::Ipv4)
continue;

if (hints.ai_family == BOOST_ASIO_OS_DEF(AF_INET6)
&& pair->RemoteHostName->Type
!= Windows::Networking::HostNameType::Ipv6)
continue;

results.values_->push_back(
basic_resolver_entry<InternetProtocol>(
typename InternetProtocol::endpoint(
ip::make_address(
boost::asio::detail::winrt_utils::string(
pair->RemoteHostName->CanonicalName)),
boost::asio::detail::winrt_utils::integer(
pair->RemoteServiceName)),
host_name, service_name));
}
}
return results;
}
# endif 
#endif 

size_type size() const BOOST_ASIO_NOEXCEPT
{
return this->values_ ? this->values_->size() : 0;
}

size_type max_size() const BOOST_ASIO_NOEXCEPT
{
return this->values_ ? this->values_->max_size() : values_type().max_size();
}

bool empty() const BOOST_ASIO_NOEXCEPT
{
return this->values_ ? this->values_->empty() : true;
}

const_iterator begin() const
{
basic_resolver_results tmp(*this);
tmp.index_ = 0;
return BOOST_ASIO_MOVE_CAST(basic_resolver_results)(tmp);
}

const_iterator end() const
{
return const_iterator();
}

const_iterator cbegin() const
{
return begin();
}

const_iterator cend() const
{
return end();
}

void swap(basic_resolver_results& that) BOOST_ASIO_NOEXCEPT
{
if (this != &that)
{
this->values_.swap(that.values_);
std::size_t index = this->index_;
this->index_ = that.index_;
that.index_ = index;
}
}

friend bool operator==(const basic_resolver_results& a,
const basic_resolver_results& b)
{
return a.equal(b);
}

friend bool operator!=(const basic_resolver_results& a,
const basic_resolver_results& b)
{
return !a.equal(b);
}

private:
typedef std::vector<basic_resolver_entry<InternetProtocol> > values_type;
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
