
#ifndef ASIO_DEADLINE_TIMER_HPP
#define ASIO_DEADLINE_TIMER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_BOOST_DATE_TIME) \
|| defined(GENERATING_DOCUMENTATION)

#include "asio/detail/socket_types.hpp" 
#include "asio/basic_deadline_timer.hpp"

#include <boost/date_time/posix_time/posix_time_types.hpp>

namespace asio {

typedef basic_deadline_timer<boost::posix_time::ptime> deadline_timer;

} 

#endif 

#endif 
