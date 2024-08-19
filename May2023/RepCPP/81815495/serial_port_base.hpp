
#ifndef ASIO_IMPL_SERIAL_PORT_BASE_HPP
#define ASIO_IMPL_SERIAL_PORT_BASE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/push_options.hpp"

namespace asio {

inline serial_port_base::baud_rate::baud_rate(unsigned int rate)
: value_(rate)
{
}

inline unsigned int serial_port_base::baud_rate::value() const
{
return value_;
}

inline serial_port_base::flow_control::type
serial_port_base::flow_control::value() const
{
return value_;
}

inline serial_port_base::parity::type serial_port_base::parity::value() const
{
return value_;
}

inline serial_port_base::stop_bits::type
serial_port_base::stop_bits::value() const
{
return value_;
}

inline unsigned int serial_port_base::character_size::value() const
{
return value_;
}

} 

#include "asio/detail/pop_options.hpp"

#endif 
