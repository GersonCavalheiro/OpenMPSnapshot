
#ifndef BOOST_INTERPROCESS_DETAIL_PTIME_TO_TIMESPEC_HPP
#define BOOST_INTERPROCESS_DETAIL_PTIME_TO_TIMESPEC_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/posix_time_types_wrk.hpp>

namespace boost {

namespace interprocess {

namespace ipcdetail {

inline timespec ptime_to_timespec (const boost::posix_time::ptime &tm)
{
const boost::posix_time::ptime epoch(boost::gregorian::date(1970,1,1));
boost::posix_time::time_duration duration  = (tm <= epoch) ? boost::posix_time::time_duration(epoch - epoch)
: boost::posix_time::time_duration(tm - epoch);
timespec ts;
ts.tv_sec  = duration.total_seconds();
ts.tv_nsec = duration.total_nanoseconds() % 1000000000;
return ts;
}

}  

}  

}  

#endif   
