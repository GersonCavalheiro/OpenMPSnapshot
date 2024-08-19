
#ifndef ASIO_IP_HOST_NAME_HPP
#define ASIO_IP_HOST_NAME_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include <string>
#include "asio/error_code.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ip {

ASIO_DECL std::string host_name();

ASIO_DECL std::string host_name(asio::error_code& ec);

} 
} 

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/ip/impl/host_name.ipp"
#endif 

#endif 
