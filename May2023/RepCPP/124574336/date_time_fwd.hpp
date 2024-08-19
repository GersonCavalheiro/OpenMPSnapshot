
#ifndef BOOST_ASIO_DETAIL_DATE_TIME_FWD_HPP
#define BOOST_ASIO_DETAIL_DATE_TIME_FWD_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

namespace boost {
namespace date_time {

template<class T, class TimeSystem>
class base_time;

} 
namespace posix_time {

class ptime;

} 
} 

#endif 
