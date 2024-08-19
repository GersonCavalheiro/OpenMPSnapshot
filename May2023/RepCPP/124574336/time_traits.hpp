
#ifndef BOOST_ASIO_TIME_TRAITS_HPP
#define BOOST_ASIO_TIME_TRAITS_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/socket_types.hpp> 

#if defined(BOOST_ASIO_HAS_BOOST_DATE_TIME) \
|| defined(GENERATING_DOCUMENTATION)

#include <boost/date_time/posix_time/posix_time_types.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {

template <typename Time>
struct time_traits;

template <>
struct time_traits<boost::posix_time::ptime>
{
typedef boost::posix_time::ptime time_type;

typedef boost::posix_time::time_duration duration_type;

static time_type now()
{
#if defined(BOOST_DATE_TIME_HAS_HIGH_PRECISION_CLOCK)
return boost::posix_time::microsec_clock::universal_time();
#else 
return boost::posix_time::second_clock::universal_time();
#endif 
}

static time_type add(const time_type& t, const duration_type& d)
{
return t + d;
}

static duration_type subtract(const time_type& t1, const time_type& t2)
{
return t1 - t2;
}

static bool less_than(const time_type& t1, const time_type& t2)
{
return t1 < t2;
}

static boost::posix_time::time_duration to_posix_duration(
const duration_type& d)
{
return d;
}
};

} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 

#endif 
