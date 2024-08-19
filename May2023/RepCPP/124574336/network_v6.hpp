
#ifndef BOOST_ASIO_IP_IMPL_NETWORK_V6_HPP
#define BOOST_ASIO_IP_IMPL_NETWORK_V6_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#if !defined(BOOST_ASIO_NO_IOSTREAM)

#include <boost/asio/detail/throw_error.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace ip {

template <typename Elem, typename Traits>
std::basic_ostream<Elem, Traits>& operator<<(
std::basic_ostream<Elem, Traits>& os, const network_v6& addr)
{
boost::system::error_code ec;
std::string s = addr.to_string(ec);
if (ec)
{
if (os.exceptions() & std::basic_ostream<Elem, Traits>::failbit)
boost::asio::detail::throw_error(ec);
else
os.setstate(std::basic_ostream<Elem, Traits>::failbit);
}
else
for (std::string::iterator i = s.begin(); i != s.end(); ++i)
os << os.widen(*i);
return os;
}

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 

#endif 
