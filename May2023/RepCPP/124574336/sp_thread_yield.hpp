#ifndef BOOST_SMART_PTR_DETAIL_SP_THREAD_YIELD_HPP_INCLUDED
#define BOOST_SMART_PTR_DETAIL_SP_THREAD_YIELD_HPP_INCLUDED


#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif


#include <boost/config.hpp>
#include <boost/config/pragma_message.hpp>

#if defined( WIN32 ) || defined( _WIN32 ) || defined( __WIN32__ ) || defined( __CYGWIN__ )

#if defined(BOOST_SP_REPORT_IMPLEMENTATION)
BOOST_PRAGMA_MESSAGE("Using Sleep(0) in sp_thread_yield")
#endif

#include <boost/smart_ptr/detail/sp_win32_sleep.hpp>

namespace boost
{

namespace detail
{

inline void sp_thread_yield()
{
Sleep( 0 );
}

} 

} 

#elif defined(BOOST_HAS_SCHED_YIELD)

#if defined(BOOST_SP_REPORT_IMPLEMENTATION)
BOOST_PRAGMA_MESSAGE("Using sched_yield() in sp_thread_yield")
#endif

#ifndef _AIX
# include <sched.h>
#else
extern "C" int sched_yield(void);
#endif

namespace boost
{

namespace detail
{

inline void sp_thread_yield()
{
sched_yield();
}

} 

} 

#else

#if defined(BOOST_SP_REPORT_IMPLEMENTATION)
BOOST_PRAGMA_MESSAGE("Using sp_thread_pause() in sp_thread_yield")
#endif

#include <boost/smart_ptr/detail/sp_thread_pause.hpp>

namespace boost
{

namespace detail
{

inline void sp_thread_yield()
{
sp_thread_pause();
}

} 

} 

#endif

#endif 
