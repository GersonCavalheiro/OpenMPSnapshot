
#ifndef BOOST_ASIO_IP_UNICAST_HPP
#define BOOST_ASIO_IP_UNICAST_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <cstddef>
#include <boost/asio/ip/detail/socket_option.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace ip {
namespace unicast {


#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined hops;
#else
typedef boost::asio::ip::detail::socket_option::unicast_hops<
BOOST_ASIO_OS_DEF(IPPROTO_IP),
BOOST_ASIO_OS_DEF(IP_TTL),
BOOST_ASIO_OS_DEF(IPPROTO_IPV6),
BOOST_ASIO_OS_DEF(IPV6_UNICAST_HOPS)> hops;
#endif

} 
} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
