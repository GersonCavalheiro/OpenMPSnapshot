
#ifndef BOOST_ASIO_SERIAL_PORT_HPP
#define BOOST_ASIO_SERIAL_PORT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if defined(BOOST_ASIO_HAS_SERIAL_PORT) \
|| defined(GENERATING_DOCUMENTATION)

#include <boost/asio/basic_serial_port.hpp>

namespace boost {
namespace asio {

typedef basic_serial_port<> serial_port;

} 
} 

#endif 

#endif 
