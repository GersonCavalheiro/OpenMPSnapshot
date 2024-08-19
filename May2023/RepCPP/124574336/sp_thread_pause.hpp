#ifndef BOOST_SMART_PTR_DETAIL_SP_THREAD_PAUSE_HPP_INCLUDED
#define BOOST_SMART_PTR_DETAIL_SP_THREAD_PAUSE_HPP_INCLUDED


#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif


#if defined(_MSC_VER) && _MSC_VER >= 1310 && ( defined(_M_IX86) || defined(_M_X64) ) && !defined(__c2__)

extern "C" void _mm_pause();

#define BOOST_SP_PAUSE _mm_pause();

#elif defined(__GNUC__) && ( defined(__i386__) || defined(__x86_64__) )

#define BOOST_SP_PAUSE __asm__ __volatile__( "rep; nop" : : : "memory" );

#else

#define BOOST_SP_PAUSE

#endif

namespace boost
{
namespace detail
{

inline void sp_thread_pause()
{
BOOST_SP_PAUSE
}

} 
} 

#undef BOOST_SP_PAUSE

#endif 
