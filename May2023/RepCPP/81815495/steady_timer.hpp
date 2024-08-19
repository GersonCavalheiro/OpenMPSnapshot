
#ifndef ASIO_STEADY_TIMER_HPP
#define ASIO_STEADY_TIMER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_CHRONO) || defined(GENERATING_DOCUMENTATION)

#include "asio/basic_waitable_timer.hpp"
#include "asio/detail/chrono.hpp"

namespace asio {


typedef basic_waitable_timer<chrono::steady_clock> steady_timer;

} 

#endif 

#endif 
