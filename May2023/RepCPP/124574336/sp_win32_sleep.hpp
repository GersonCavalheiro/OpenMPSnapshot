#ifndef BOOST_SMART_PTR_DETAIL_SP_WIN32_SLEEP_HPP_INCLUDED
#define BOOST_SMART_PTR_DETAIL_SP_WIN32_SLEEP_HPP_INCLUDED


#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif


#if defined( BOOST_USE_WINDOWS_H )
# include <windows.h>
#endif

namespace boost
{
namespace detail
{

#if !defined( BOOST_USE_WINDOWS_H )

#if defined(__clang__) && defined(__x86_64__)
# define BOOST_SP_STDCALL
#else
# define BOOST_SP_STDCALL __stdcall
#endif

#if defined(__LP64__) 
extern "C" __declspec(dllimport) void BOOST_SP_STDCALL Sleep( unsigned int ms );
#else
extern "C" __declspec(dllimport) void BOOST_SP_STDCALL Sleep( unsigned long ms );
#endif

#undef BOOST_SP_STDCALL

#endif 

} 
} 

#endif 
