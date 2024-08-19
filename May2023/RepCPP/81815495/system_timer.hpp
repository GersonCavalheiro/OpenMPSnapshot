
#ifndef ASIO_SYSTEM_TIMER_HPP
#define ASIO_SYSTEM_TIMER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_CHRONO) || defined(GENERATING_DOCUMENTATION)

#include "asio/basic_waitable_timer.hpp"
#include "asio/detail/chrono.hpp"

namespace asio {


typedef basic_waitable_timer<chrono::system_clock> system_timer;

} 

#endif 

#endif 
