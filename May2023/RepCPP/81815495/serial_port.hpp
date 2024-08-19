
#ifndef ASIO_SERIAL_PORT_HPP
#define ASIO_SERIAL_PORT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_SERIAL_PORT) \
|| defined(GENERATING_DOCUMENTATION)

#include "asio/basic_serial_port.hpp"

namespace asio {

typedef basic_serial_port<> serial_port;

} 

#endif 

#endif 
