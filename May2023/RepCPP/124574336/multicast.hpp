
#ifndef BOOST_ASIO_IP_MULTICAST_HPP
#define BOOST_ASIO_IP_MULTICAST_HPP

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
namespace multicast {


#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined join_group;
#else
typedef boost::asio::ip::detail::socket_option::multicast_request<
BOOST_ASIO_OS_DEF(IPPROTO_IP),
BOOST_ASIO_OS_DEF(IP_ADD_MEMBERSHIP),
BOOST_ASIO_OS_DEF(IPPROTO_IPV6),
BOOST_ASIO_OS_DEF(IPV6_JOIN_GROUP)> join_group;
#endif


#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined leave_group;
#else
typedef boost::asio::ip::detail::socket_option::multicast_request<
BOOST_ASIO_OS_DEF(IPPROTO_IP),
BOOST_ASIO_OS_DEF(IP_DROP_MEMBERSHIP),
BOOST_ASIO_OS_DEF(IPPROTO_IPV6),
BOOST_ASIO_OS_DEF(IPV6_LEAVE_GROUP)> leave_group;
#endif


#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined outbound_interface;
#else
typedef boost::asio::ip::detail::socket_option::network_interface<
BOOST_ASIO_OS_DEF(IPPROTO_IP),
BOOST_ASIO_OS_DEF(IP_MULTICAST_IF),
BOOST_ASIO_OS_DEF(IPPROTO_IPV6),
BOOST_ASIO_OS_DEF(IPV6_MULTICAST_IF)> outbound_interface;
#endif


#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined hops;
#else
typedef boost::asio::ip::detail::socket_option::multicast_hops<
BOOST_ASIO_OS_DEF(IPPROTO_IP),
BOOST_ASIO_OS_DEF(IP_MULTICAST_TTL),
BOOST_ASIO_OS_DEF(IPPROTO_IPV6),
BOOST_ASIO_OS_DEF(IPV6_MULTICAST_HOPS)> hops;
#endif


#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined enable_loopback;
#else
typedef boost::asio::ip::detail::socket_option::multicast_enable_loopback<
BOOST_ASIO_OS_DEF(IPPROTO_IP),
BOOST_ASIO_OS_DEF(IP_MULTICAST_LOOP),
BOOST_ASIO_OS_DEF(IPPROTO_IPV6),
BOOST_ASIO_OS_DEF(IPV6_MULTICAST_LOOP)> enable_loopback;
#endif

} 
} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
