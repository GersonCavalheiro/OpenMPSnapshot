
#ifndef ASIO_IP_V6_ONLY_HPP
#define ASIO_IP_V6_ONLY_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/detail/socket_option.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ip {


#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined v6_only;
#elif defined(IPV6_V6ONLY)
typedef asio::detail::socket_option::boolean<
IPPROTO_IPV6, IPV6_V6ONLY> v6_only;
#else
typedef asio::detail::socket_option::boolean<
asio::detail::custom_socket_option_level,
asio::detail::always_fail_option> v6_only;
#endif

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 
