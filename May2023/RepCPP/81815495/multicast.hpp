
#ifndef ASIO_IP_MULTICAST_HPP
#define ASIO_IP_MULTICAST_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include <cstddef>
#include "asio/ip/detail/socket_option.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ip {
namespace multicast {


#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined join_group;
#else
typedef asio::ip::detail::socket_option::multicast_request<
ASIO_OS_DEF(IPPROTO_IP),
ASIO_OS_DEF(IP_ADD_MEMBERSHIP),
ASIO_OS_DEF(IPPROTO_IPV6),
ASIO_OS_DEF(IPV6_JOIN_GROUP)> join_group;
#endif


#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined leave_group;
#else
typedef asio::ip::detail::socket_option::multicast_request<
ASIO_OS_DEF(IPPROTO_IP),
ASIO_OS_DEF(IP_DROP_MEMBERSHIP),
ASIO_OS_DEF(IPPROTO_IPV6),
ASIO_OS_DEF(IPV6_LEAVE_GROUP)> leave_group;
#endif


#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined outbound_interface;
#else
typedef asio::ip::detail::socket_option::network_interface<
ASIO_OS_DEF(IPPROTO_IP),
ASIO_OS_DEF(IP_MULTICAST_IF),
ASIO_OS_DEF(IPPROTO_IPV6),
ASIO_OS_DEF(IPV6_MULTICAST_IF)> outbound_interface;
#endif


#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined hops;
#else
typedef asio::ip::detail::socket_option::multicast_hops<
ASIO_OS_DEF(IPPROTO_IP),
ASIO_OS_DEF(IP_MULTICAST_TTL),
ASIO_OS_DEF(IPPROTO_IPV6),
ASIO_OS_DEF(IPV6_MULTICAST_HOPS)> hops;
#endif


#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined enable_loopback;
#else
typedef asio::ip::detail::socket_option::multicast_enable_loopback<
ASIO_OS_DEF(IPPROTO_IP),
ASIO_OS_DEF(IP_MULTICAST_LOOP),
ASIO_OS_DEF(IPPROTO_IPV6),
ASIO_OS_DEF(IPV6_MULTICAST_LOOP)> enable_loopback;
#endif

} 
} 
} 

#include "asio/detail/pop_options.hpp"

#endif 
