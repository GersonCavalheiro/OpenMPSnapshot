

#ifndef BOOST_WINAPI_GET_PROC_ADDRESS_HPP_INCLUDED_
#define BOOST_WINAPI_GET_PROC_ADDRESS_HPP_INCLUDED_

#include <boost/winapi/basic_types.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

#if BOOST_WINAPI_PARTITION_DESKTOP || BOOST_WINAPI_PARTITION_SYSTEM

#include <boost/winapi/detail/header.hpp>

#if !defined(BOOST_USE_WINDOWS_H)
namespace boost { namespace winapi {
#ifdef _WIN64
typedef INT_PTR_ (BOOST_WINAPI_WINAPI_CC *FARPROC_)();
typedef INT_PTR_ (BOOST_WINAPI_WINAPI_CC *NEARPROC_)();
typedef INT_PTR_ (BOOST_WINAPI_WINAPI_CC *PROC_)();
#else
typedef int (BOOST_WINAPI_WINAPI_CC *FARPROC_)();
typedef int (BOOST_WINAPI_WINAPI_CC *NEARPROC_)();
typedef int (BOOST_WINAPI_WINAPI_CC *PROC_)();
#endif 
}} 

extern "C" {
#if !defined(UNDER_CE)
BOOST_WINAPI_IMPORT boost::winapi::FARPROC_ BOOST_WINAPI_WINAPI_CC
GetProcAddress(boost::winapi::HMODULE_ hModule, boost::winapi::LPCSTR_ lpProcName);
#else
BOOST_WINAPI_IMPORT_EXCEPT_WM boost::winapi::FARPROC_ BOOST_WINAPI_WINAPI_CC
GetProcAddressA(boost::winapi::HMODULE_ hModule, boost::winapi::LPCSTR_ lpProcName);
BOOST_WINAPI_IMPORT_EXCEPT_WM boost::winapi::FARPROC_ BOOST_WINAPI_WINAPI_CC
GetProcAddressW(boost::winapi::HMODULE_ hModule, boost::winapi::LPCWSTR_ lpProcName);
#endif
} 
#endif 

namespace boost {
namespace winapi {

#if defined(BOOST_USE_WINDOWS_H)
typedef ::FARPROC FARPROC_;
typedef ::NEARPROC NEARPROC_;
typedef ::PROC PROC_;
#endif 

#if !defined(UNDER_CE)
using ::GetProcAddress;
#else
using ::GetProcAddressA;
using ::GetProcAddressW;
#endif

BOOST_FORCEINLINE FARPROC_ get_proc_address(HMODULE_ hModule, LPCSTR_ lpProcName)
{
#if !defined(UNDER_CE)
return ::GetProcAddress(hModule, lpProcName);
#else
return ::GetProcAddressA(hModule, lpProcName);
#endif
}

} 
} 

#include <boost/winapi/detail/footer.hpp>

#endif 
#endif 
