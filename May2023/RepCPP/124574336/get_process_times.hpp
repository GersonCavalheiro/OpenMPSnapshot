

#ifndef BOOST_WINAPI_GET_PROCESS_TIMES_HPP_INCLUDED_
#define BOOST_WINAPI_GET_PROCESS_TIMES_HPP_INCLUDED_

#include <boost/winapi/config.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

#if !defined( UNDER_CE )

#if BOOST_WINAPI_PARTITION_APP_SYSTEM

#include <boost/winapi/basic_types.hpp>
#include <boost/winapi/time.hpp>
#include <boost/winapi/detail/header.hpp>

#if !defined( BOOST_USE_WINDOWS_H )
extern "C" {
BOOST_WINAPI_IMPORT boost::winapi::BOOL_ BOOST_WINAPI_WINAPI_CC
GetProcessTimes(
boost::winapi::HANDLE_ hProcess,
::_FILETIME* lpCreationTime,
::_FILETIME* lpExitTime,
::_FILETIME* lpKernelTime,
::_FILETIME* lpUserTime);
}
#endif

namespace boost {
namespace winapi {

BOOST_FORCEINLINE BOOL_ GetProcessTimes(
HANDLE_ hProcess,
LPFILETIME_ lpCreationTime,
LPFILETIME_ lpExitTime,
LPFILETIME_ lpKernelTime,
LPFILETIME_ lpUserTime)
{
return ::GetProcessTimes(
hProcess,
reinterpret_cast< ::_FILETIME* >(lpCreationTime),
reinterpret_cast< ::_FILETIME* >(lpExitTime),
reinterpret_cast< ::_FILETIME* >(lpKernelTime),
reinterpret_cast< ::_FILETIME* >(lpUserTime));
}

}
}

#include <boost/winapi/detail/footer.hpp>

#endif 
#endif 
#endif 