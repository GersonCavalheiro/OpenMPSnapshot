#ifndef BOOST_SMART_PTR_DETAIL_SP_THREAD_SLEEP_HPP_INCLUDED
#define BOOST_SMART_PTR_DETAIL_SP_THREAD_SLEEP_HPP_INCLUDED


#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif


#include <boost/config.hpp>
#include <boost/config/pragma_message.hpp>

#if defined( WIN32 ) || defined( _WIN32 ) || defined( __WIN32__ ) || defined( __CYGWIN__ )

#if defined(BOOST_SP_REPORT_IMPLEMENTATION)
BOOST_PRAGMA_MESSAGE("Using Sleep(1) in sp_thread_sleep")
#endif

#include <boost/smart_ptr/detail/sp_win32_sleep.hpp>

namespace boost
{

namespace detail
{

inline void sp_thread_sleep()
{
Sleep( 1 );
}

} 

} 

#elif defined(BOOST_HAS_NANOSLEEP)

#if defined(BOOST_SP_REPORT_IMPLEMENTATION)
BOOST_PRAGMA_MESSAGE("Using nanosleep() in sp_thread_sleep")
#endif

#include <time.h>

namespace boost
{

namespace detail
{

inline void sp_thread_sleep()
{
struct timespec rqtp = { 0, 0 };


rqtp.tv_sec = 0;
rqtp.tv_nsec = 1000;

nanosleep( &rqtp, 0 );
}

} 

} 

#else

#if defined(BOOST_SP_REPORT_IMPLEMENTATION)
BOOST_PRAGMA_MESSAGE("Using sp_thread_yield() in sp_thread_sleep")
#endif

#include <boost/smart_ptr/detail/sp_thread_yield.hpp>

namespace boost
{

namespace detail
{

inline void sp_thread_sleep()
{
sp_thread_yield();
}

} 

} 

#endif

#endif 
