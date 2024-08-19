
#ifndef ASIO_IP_UNICAST_HPP
#define ASIO_IP_UNICAST_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include <cstddef>
#include "asio/ip/detail/socket_option.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ip {
namespace unicast {


#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined hops;
#else
typedef asio::ip::detail::socket_option::unicast_hops<
ASIO_OS_DEF(IPPROTO_IP),
ASIO_OS_DEF(IP_TTL),
ASIO_OS_DEF(IPPROTO_IPV6),
ASIO_OS_DEF(IPV6_UNICAST_HOPS)> hops;
#endif

} 
} 
} 

#include "asio/detail/pop_options.hpp"

#endif 
