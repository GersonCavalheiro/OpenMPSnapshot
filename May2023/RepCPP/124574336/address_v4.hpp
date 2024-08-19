
#ifndef BOOST_ASIO_IP_IMPL_ADDRESS_V4_HPP
#define BOOST_ASIO_IP_IMPL_ADDRESS_V4_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#if !defined(BOOST_ASIO_NO_IOSTREAM)

#include <boost/asio/detail/throw_error.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace ip {

#if !defined(BOOST_ASIO_NO_DEPRECATED)

inline address_v4 address_v4::from_string(const char* str)
{
return boost::asio::ip::make_address_v4(str);
}

inline address_v4 address_v4::from_string(
const char* str, boost::system::error_code& ec)
{
return boost::asio::ip::make_address_v4(str, ec);
}

inline address_v4 address_v4::from_string(const std::string& str)
{
return boost::asio::ip::make_address_v4(str);
}

inline address_v4 address_v4::from_string(
const std::string& str, boost::system::error_code& ec)
{
return boost::asio::ip::make_address_v4(str, ec);
}

#endif 

template <typename Elem, typename Traits>
std::basic_ostream<Elem, Traits>& operator<<(
std::basic_ostream<Elem, Traits>& os, const address_v4& addr)
{
return os << addr.to_string().c_str();
}

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 

#endif 
