

#ifndef BOOST_WINAPI_TIMERS_HPP_INCLUDED_
#define BOOST_WINAPI_TIMERS_HPP_INCLUDED_

#include <boost/winapi/basic_types.hpp>
#include <boost/winapi/detail/header.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

#if !defined( BOOST_USE_WINDOWS_H )
extern "C" {
BOOST_WINAPI_IMPORT_EXCEPT_WM boost::winapi::BOOL_ BOOST_WINAPI_WINAPI_CC
QueryPerformanceCounter(::_LARGE_INTEGER* lpPerformanceCount);

BOOST_WINAPI_IMPORT_EXCEPT_WM boost::winapi::BOOL_ BOOST_WINAPI_WINAPI_CC
QueryPerformanceFrequency(::_LARGE_INTEGER* lpFrequency);
} 
#endif


namespace boost {
namespace winapi {

BOOST_FORCEINLINE BOOL_ QueryPerformanceCounter(LARGE_INTEGER_* lpPerformanceCount)
{
return ::QueryPerformanceCounter(reinterpret_cast< ::_LARGE_INTEGER* >(lpPerformanceCount));
}

BOOST_FORCEINLINE BOOL_ QueryPerformanceFrequency(LARGE_INTEGER_* lpFrequency)
{
return ::QueryPerformanceFrequency(reinterpret_cast< ::_LARGE_INTEGER* >(lpFrequency));
}

}
}

#include <boost/winapi/detail/footer.hpp>

#endif 
