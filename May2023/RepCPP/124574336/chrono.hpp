
#ifndef BOOST_ASIO_DETAIL_CHRONO_HPP
#define BOOST_ASIO_DETAIL_CHRONO_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if defined(BOOST_ASIO_HAS_STD_CHRONO)
# include <chrono>
#elif defined(BOOST_ASIO_HAS_BOOST_CHRONO)
# include <boost/chrono/system_clocks.hpp>
#endif 

namespace boost {
namespace asio {
namespace chrono {

#if defined(BOOST_ASIO_HAS_STD_CHRONO)
using std::chrono::duration;
using std::chrono::time_point;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;
using std::chrono::microseconds;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::minutes;
using std::chrono::hours;
using std::chrono::time_point_cast;
#if defined(BOOST_ASIO_HAS_STD_CHRONO_MONOTONIC_CLOCK)
typedef std::chrono::monotonic_clock steady_clock;
#else 
using std::chrono::steady_clock;
#endif 
using std::chrono::system_clock;
using std::chrono::high_resolution_clock;
#elif defined(BOOST_ASIO_HAS_BOOST_CHRONO)
using boost::chrono::duration;
using boost::chrono::time_point;
using boost::chrono::duration_cast;
using boost::chrono::nanoseconds;
using boost::chrono::microseconds;
using boost::chrono::milliseconds;
using boost::chrono::seconds;
using boost::chrono::minutes;
using boost::chrono::hours;
using boost::chrono::time_point_cast;
using boost::chrono::system_clock;
using boost::chrono::steady_clock;
using boost::chrono::high_resolution_clock;
#endif 

} 
} 
} 

#endif 
