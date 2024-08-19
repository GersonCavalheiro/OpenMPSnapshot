
#ifndef ASIO_LOCAL_DETAIL_ENDPOINT_HPP
#define ASIO_LOCAL_DETAIL_ENDPOINT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_LOCAL_SOCKETS)

#include <cstddef>
#include <string>
#include "asio/detail/socket_types.hpp"
#include "asio/detail/string_view.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace local {
namespace detail {

class endpoint
{
public:
ASIO_DECL endpoint();

ASIO_DECL endpoint(const char* path_name);

ASIO_DECL endpoint(const std::string& path_name);

#if defined(ASIO_HAS_STRING_VIEW)
ASIO_DECL endpoint(string_view path_name);
#endif 

endpoint(const endpoint& other)
: data_(other.data_),
path_length_(other.path_length_)
{
}

endpoint& operator=(const endpoint& other)
{
data_ = other.data_;
path_length_ = other.path_length_;
return *this;
}

asio::detail::socket_addr_type* data()
{
return &data_.base;
}

const asio::detail::socket_addr_type* data() const
{
return &data_.base;
}

std::size_t size() const
{
return path_length_
+ offsetof(asio::detail::sockaddr_un_type, sun_path);
}

ASIO_DECL void resize(std::size_t size);

std::size_t capacity() const
{
return sizeof(asio::detail::sockaddr_un_type);
}

ASIO_DECL std::string path() const;

ASIO_DECL void path(const char* p);

ASIO_DECL void path(const std::string& p);

ASIO_DECL friend bool operator==(
const endpoint& e1, const endpoint& e2);

ASIO_DECL friend bool operator<(
const endpoint& e1, const endpoint& e2);

private:
union data_union
{
asio::detail::socket_addr_type base;
asio::detail::sockaddr_un_type local;
} data_;

std::size_t path_length_;

ASIO_DECL void init(const char* path, std::size_t path_length);
};

} 
} 
} 

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/local/detail/impl/endpoint.ipp"
#endif 

#endif 

#endif 
