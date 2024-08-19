
#ifndef ASIO_IP_IMPL_ADDRESS_V6_HPP
#define ASIO_IP_IMPL_ADDRESS_V6_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#if !defined(ASIO_NO_IOSTREAM)

#include "asio/detail/throw_error.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ip {

#if !defined(ASIO_NO_DEPRECATED)

inline address_v6 address_v6::from_string(const char* str)
{
return asio::ip::make_address_v6(str);
}

inline address_v6 address_v6::from_string(
const char* str, asio::error_code& ec)
{
return asio::ip::make_address_v6(str, ec);
}

inline address_v6 address_v6::from_string(const std::string& str)
{
return asio::ip::make_address_v6(str);
}

inline address_v6 address_v6::from_string(
const std::string& str, asio::error_code& ec)
{
return asio::ip::make_address_v6(str, ec);
}

#endif 

template <typename Elem, typename Traits>
std::basic_ostream<Elem, Traits>& operator<<(
std::basic_ostream<Elem, Traits>& os, const address_v6& addr)
{
return os << addr.to_string().c_str();
}

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 

#endif 
