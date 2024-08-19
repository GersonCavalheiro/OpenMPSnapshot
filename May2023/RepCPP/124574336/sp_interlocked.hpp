#ifndef BOOST_SMART_PTR_DETAIL_SP_INTERLOCKED_HPP_INCLUDED
#define BOOST_SMART_PTR_DETAIL_SP_INTERLOCKED_HPP_INCLUDED


#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif


#include <boost/config.hpp>


#if defined( BOOST_MSVC ) && BOOST_MSVC >= 1600

# define BOOST_SP_HAS_INTRIN_H

#elif defined( __MINGW64_VERSION_MAJOR )

# define BOOST_SP_HAS_INTRIN_H

#elif defined( __LP64__ )

# define BOOST_SP_HAS_INTRIN_H

#elif defined( BOOST_INTEL_WIN ) && defined( _CPPLIB_VER ) && _CPPLIB_VER >= 520

# define BOOST_SP_HAS_INTRIN_H

#elif defined( __clang__ ) && defined( _MSC_VER ) && defined( _CPPLIB_VER ) && _CPPLIB_VER >= 520

# define BOOST_SP_HAS_INTRIN_H

#endif

#if defined( BOOST_USE_WINDOWS_H )

# include <windows.h>

# define BOOST_SP_INTERLOCKED_INCREMENT InterlockedIncrement
# define BOOST_SP_INTERLOCKED_DECREMENT InterlockedDecrement
# define BOOST_SP_INTERLOCKED_COMPARE_EXCHANGE InterlockedCompareExchange
# define BOOST_SP_INTERLOCKED_EXCHANGE InterlockedExchange
# define BOOST_SP_INTERLOCKED_EXCHANGE_ADD InterlockedExchangeAdd

#elif defined( BOOST_USE_INTRIN_H ) || defined( BOOST_SP_HAS_INTRIN_H )

#include <intrin.h>

# define BOOST_SP_INTERLOCKED_INCREMENT _InterlockedIncrement
# define BOOST_SP_INTERLOCKED_DECREMENT _InterlockedDecrement
# define BOOST_SP_INTERLOCKED_COMPARE_EXCHANGE _InterlockedCompareExchange
# define BOOST_SP_INTERLOCKED_EXCHANGE _InterlockedExchange
# define BOOST_SP_INTERLOCKED_EXCHANGE_ADD _InterlockedExchangeAdd

#elif defined( _WIN32_WCE )

#if _WIN32_WCE >= 0x600

extern "C" long __cdecl _InterlockedIncrement( long volatile * );
extern "C" long __cdecl _InterlockedDecrement( long volatile * );
extern "C" long __cdecl _InterlockedCompareExchange( long volatile *, long, long );
extern "C" long __cdecl _InterlockedExchange( long volatile *, long );
extern "C" long __cdecl _InterlockedExchangeAdd( long volatile *, long );

# define BOOST_SP_INTERLOCKED_INCREMENT _InterlockedIncrement
# define BOOST_SP_INTERLOCKED_DECREMENT _InterlockedDecrement
# define BOOST_SP_INTERLOCKED_COMPARE_EXCHANGE _InterlockedCompareExchange
# define BOOST_SP_INTERLOCKED_EXCHANGE _InterlockedExchange
# define BOOST_SP_INTERLOCKED_EXCHANGE_ADD _InterlockedExchangeAdd

#else


extern "C" long __cdecl InterlockedIncrement( long* );
extern "C" long __cdecl InterlockedDecrement( long* );
extern "C" long __cdecl InterlockedCompareExchange( long*, long, long );
extern "C" long __cdecl InterlockedExchange( long*, long );
extern "C" long __cdecl InterlockedExchangeAdd( long*, long );

# define BOOST_SP_INTERLOCKED_INCREMENT InterlockedIncrement
# define BOOST_SP_INTERLOCKED_DECREMENT InterlockedDecrement
# define BOOST_SP_INTERLOCKED_COMPARE_EXCHANGE InterlockedCompareExchange
# define BOOST_SP_INTERLOCKED_EXCHANGE InterlockedExchange
# define BOOST_SP_INTERLOCKED_EXCHANGE_ADD InterlockedExchangeAdd

#endif

#elif defined( BOOST_MSVC ) || defined( BOOST_INTEL_WIN )

#if defined( __CLRCALL_PURE_OR_CDECL )

extern "C" long __CLRCALL_PURE_OR_CDECL _InterlockedIncrement( long volatile * );
extern "C" long __CLRCALL_PURE_OR_CDECL _InterlockedDecrement( long volatile * );
extern "C" long __CLRCALL_PURE_OR_CDECL _InterlockedCompareExchange( long volatile *, long, long );
extern "C" long __CLRCALL_PURE_OR_CDECL _InterlockedExchange( long volatile *, long );
extern "C" long __CLRCALL_PURE_OR_CDECL _InterlockedExchangeAdd( long volatile *, long );

#else

extern "C" long __cdecl _InterlockedIncrement( long volatile * );
extern "C" long __cdecl _InterlockedDecrement( long volatile * );
extern "C" long __cdecl _InterlockedCompareExchange( long volatile *, long, long );
extern "C" long __cdecl _InterlockedExchange( long volatile *, long );
extern "C" long __cdecl _InterlockedExchangeAdd( long volatile *, long );

# if defined( BOOST_MSVC ) && BOOST_MSVC == 1310
#  pragma intrinsic( _InterlockedIncrement )
#  pragma intrinsic( _InterlockedDecrement )
#  pragma intrinsic( _InterlockedCompareExchange )
#  pragma intrinsic( _InterlockedExchange )
#  pragma intrinsic( _InterlockedExchangeAdd )
# endif

#endif

# define BOOST_SP_INTERLOCKED_INCREMENT _InterlockedIncrement
# define BOOST_SP_INTERLOCKED_DECREMENT _InterlockedDecrement
# define BOOST_SP_INTERLOCKED_COMPARE_EXCHANGE _InterlockedCompareExchange
# define BOOST_SP_INTERLOCKED_EXCHANGE _InterlockedExchange
# define BOOST_SP_INTERLOCKED_EXCHANGE_ADD _InterlockedExchangeAdd

#elif defined( WIN32 ) || defined( _WIN32 ) || defined( __WIN32__ ) || defined( __CYGWIN__ )

namespace boost
{

namespace detail
{

extern "C" __declspec(dllimport) long __stdcall InterlockedIncrement( long volatile * );
extern "C" __declspec(dllimport) long __stdcall InterlockedDecrement( long volatile * );
extern "C" __declspec(dllimport) long __stdcall InterlockedCompareExchange( long volatile *, long, long );
extern "C" __declspec(dllimport) long __stdcall InterlockedExchange( long volatile *, long );
extern "C" __declspec(dllimport) long __stdcall InterlockedExchangeAdd( long volatile *, long );

} 

} 

# define BOOST_SP_INTERLOCKED_INCREMENT ::boost::detail::InterlockedIncrement
# define BOOST_SP_INTERLOCKED_DECREMENT ::boost::detail::InterlockedDecrement
# define BOOST_SP_INTERLOCKED_COMPARE_EXCHANGE ::boost::detail::InterlockedCompareExchange
# define BOOST_SP_INTERLOCKED_EXCHANGE ::boost::detail::InterlockedExchange
# define BOOST_SP_INTERLOCKED_EXCHANGE_ADD ::boost::detail::InterlockedExchangeAdd

#else

# error "Interlocked intrinsics not available"

#endif

#endif 
