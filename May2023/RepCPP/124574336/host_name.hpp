
#ifndef BOOST_ASIO_IP_HOST_NAME_HPP
#define BOOST_ASIO_IP_HOST_NAME_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <string>
#include <boost/system/error_code.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace ip {

BOOST_ASIO_DECL std::string host_name();

BOOST_ASIO_DECL std::string host_name(boost::system::error_code& ec);

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#if defined(BOOST_ASIO_HEADER_ONLY)
# include <boost/asio/ip/impl/host_name.ipp>
#endif 

#endif 
