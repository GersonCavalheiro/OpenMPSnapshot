
#ifndef BOOST_ASIO_DEADLINE_TIMER_HPP
#define BOOST_ASIO_DEADLINE_TIMER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if defined(BOOST_ASIO_HAS_BOOST_DATE_TIME) \
|| defined(GENERATING_DOCUMENTATION)

#include <boost/asio/detail/socket_types.hpp> 
#include <boost/asio/basic_deadline_timer.hpp>

#include <boost/date_time/posix_time/posix_time_types.hpp>

namespace boost {
namespace asio {

typedef basic_deadline_timer<boost::posix_time::ptime> deadline_timer;

} 
} 

#endif 

#endif 
