
#ifndef BOOST_ASIO_LOCAL_DETAIL_ENDPOINT_HPP
#define BOOST_ASIO_LOCAL_DETAIL_ENDPOINT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if defined(BOOST_ASIO_HAS_LOCAL_SOCKETS)

#include <cstddef>
#include <string>
#include <boost/asio/detail/socket_types.hpp>
#include <boost/asio/detail/string_view.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace local {
namespace detail {

class endpoint
{
public:
BOOST_ASIO_DECL endpoint();

BOOST_ASIO_DECL endpoint(const char* path_name);

BOOST_ASIO_DECL endpoint(const std::string& path_name);

#if defined(BOOST_ASIO_HAS_STRING_VIEW)
BOOST_ASIO_DECL endpoint(string_view path_name);
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

boost::asio::detail::socket_addr_type* data()
{
return &data_.base;
}

const boost::asio::detail::socket_addr_type* data() const
{
return &data_.base;
}

std::size_t size() const
{
return path_length_
+ offsetof(boost::asio::detail::sockaddr_un_type, sun_path);
}

BOOST_ASIO_DECL void resize(std::size_t size);

std::size_t capacity() const
{
return sizeof(boost::asio::detail::sockaddr_un_type);
}

BOOST_ASIO_DECL std::string path() const;

BOOST_ASIO_DECL void path(const char* p);

BOOST_ASIO_DECL void path(const std::string& p);

BOOST_ASIO_DECL friend bool operator==(
const endpoint& e1, const endpoint& e2);

BOOST_ASIO_DECL friend bool operator<(
const endpoint& e1, const endpoint& e2);

private:
union data_union
{
boost::asio::detail::socket_addr_type base;
boost::asio::detail::sockaddr_un_type local;
} data_;

std::size_t path_length_;

BOOST_ASIO_DECL void init(const char* path, std::size_t path_length);
};

} 
} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#if defined(BOOST_ASIO_HEADER_ONLY)
# include <boost/asio/local/detail/impl/endpoint.ipp>
#endif 

#endif 

#endif 
