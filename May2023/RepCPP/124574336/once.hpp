#ifndef BOOST_THREAD_ONCE_HPP
#define BOOST_THREAD_ONCE_HPP


#include <boost/thread/detail/config.hpp>

#ifdef BOOST_MSVC
# pragma warning(push)
# pragma warning(disable: 4702) 
#endif

#include <boost/thread/detail/platform.hpp>
#if defined(BOOST_THREAD_PLATFORM_WIN32)
#include <boost/thread/win32/once.hpp>
#elif defined(BOOST_THREAD_PLATFORM_PTHREAD)
#if defined BOOST_THREAD_ONCE_FAST_EPOCH
#include <boost/thread/pthread/once.hpp>
#elif defined BOOST_THREAD_ONCE_ATOMIC
#include <boost/thread/pthread/once_atomic.hpp>
#else
#error "Once Not Implemented"
#endif
#else
#error "Boost threads unavailable on this platform"
#endif

#include <boost/config/abi_prefix.hpp>

namespace boost
{
template<typename Function>
inline void call_once(Function func,once_flag& flag)
{
call_once(flag,func);
}
}

#include <boost/config/abi_suffix.hpp>

#ifdef BOOST_MSVC
# pragma warning(pop)
#endif

#endif
