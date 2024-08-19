
#ifndef ASIO_LOCAL_BASIC_ENDPOINT_HPP
#define ASIO_LOCAL_BASIC_ENDPOINT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_LOCAL_SOCKETS) \
|| defined(GENERATING_DOCUMENTATION)

#include "asio/local/detail/endpoint.hpp"

#if !defined(ASIO_NO_IOSTREAM)
# include <iosfwd>
#endif 

#include "asio/detail/push_options.hpp"

namespace asio {
namespace local {


template <typename Protocol>
class basic_endpoint
{
public:
typedef Protocol protocol_type;

#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined data_type;
#else
typedef asio::detail::socket_addr_type data_type;
#endif

basic_endpoint() ASIO_NOEXCEPT
{
}

basic_endpoint(const char* path_name)
: impl_(path_name)
{
}

basic_endpoint(const std::string& path_name)
: impl_(path_name)
{
}

#if defined(ASIO_HAS_STRING_VIEW)
basic_endpoint(string_view path_name)
: impl_(path_name)
{
}
#endif 

basic_endpoint(const basic_endpoint& other)
: impl_(other.impl_)
{
}

#if defined(ASIO_HAS_MOVE)
basic_endpoint(basic_endpoint&& other)
: impl_(other.impl_)
{
}
#endif 

basic_endpoint& operator=(const basic_endpoint& other)
{
impl_ = other.impl_;
return *this;
}

#if defined(ASIO_HAS_MOVE)
basic_endpoint& operator=(basic_endpoint&& other)
{
impl_ = other.impl_;
return *this;
}
#endif 

protocol_type protocol() const
{
return protocol_type();
}

data_type* data()
{
return impl_.data();
}

const data_type* data() const
{
return impl_.data();
}

std::size_t size() const
{
return impl_.size();
}

void resize(std::size_t new_size)
{
impl_.resize(new_size);
}

std::size_t capacity() const
{
return impl_.capacity();
}

std::string path() const
{
return impl_.path();
}

void path(const char* p)
{
impl_.path(p);
}

void path(const std::string& p)
{
impl_.path(p);
}

friend bool operator==(const basic_endpoint<Protocol>& e1,
const basic_endpoint<Protocol>& e2)
{
return e1.impl_ == e2.impl_;
}

friend bool operator!=(const basic_endpoint<Protocol>& e1,
const basic_endpoint<Protocol>& e2)
{
return !(e1.impl_ == e2.impl_);
}

friend bool operator<(const basic_endpoint<Protocol>& e1,
const basic_endpoint<Protocol>& e2)
{
return e1.impl_ < e2.impl_;
}

friend bool operator>(const basic_endpoint<Protocol>& e1,
const basic_endpoint<Protocol>& e2)
{
return e2.impl_ < e1.impl_;
}

friend bool operator<=(const basic_endpoint<Protocol>& e1,
const basic_endpoint<Protocol>& e2)
{
return !(e2 < e1);
}

friend bool operator>=(const basic_endpoint<Protocol>& e1,
const basic_endpoint<Protocol>& e2)
{
return !(e1 < e2);
}

private:
asio::local::detail::endpoint impl_;
};


template <typename Elem, typename Traits, typename Protocol>
std::basic_ostream<Elem, Traits>& operator<<(
std::basic_ostream<Elem, Traits>& os,
const basic_endpoint<Protocol>& endpoint)
{
os << endpoint.path();
return os;
}

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 

#endif 
