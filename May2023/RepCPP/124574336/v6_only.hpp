
#ifndef BOOST_ASIO_IP_V6_ONLY_HPP
#define BOOST_ASIO_IP_V6_ONLY_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/detail/socket_option.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace ip {


#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined v6_only;
#elif defined(IPV6_V6ONLY)
typedef boost::asio::detail::socket_option::boolean<
IPPROTO_IPV6, IPV6_V6ONLY> v6_only;
#else
typedef boost::asio::detail::socket_option::boolean<
boost::asio::detail::custom_socket_option_level,
boost::asio::detail::always_fail_option> v6_only;
#endif

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
