

#ifndef BOOST_WINAPI_LOCAL_MEMORY_HPP_INCLUDED_
#define BOOST_WINAPI_LOCAL_MEMORY_HPP_INCLUDED_

#include <boost/winapi/basic_types.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

#if BOOST_WINAPI_PARTITION_APP_SYSTEM

#include <boost/winapi/detail/header.hpp>

#if !defined( BOOST_USE_WINDOWS_H )
namespace boost { namespace winapi {
typedef HANDLE_ HLOCAL_;
}}

extern "C" {

#if defined (_WIN32_WCE )
BOOST_WINAPI_IMPORT_EXCEPT_WM boost::winapi::HLOCAL_ BOOST_WINAPI_WINAPI_CC
LocalAlloc(
boost::winapi::UINT_ uFlags,
boost::winapi::UINT_ uBytes);

BOOST_WINAPI_IMPORT_EXCEPT_WM boost::winapi::HLOCAL_ BOOST_WINAPI_WINAPI_CC
LocalReAlloc(
boost::winapi::HLOCAL_ hMem,
boost::winapi::UINT_ uBytes,
boost::winapi::UINT_ uFlags);
#else
BOOST_WINAPI_IMPORT_EXCEPT_WM boost::winapi::HLOCAL_ BOOST_WINAPI_WINAPI_CC
LocalAlloc(
boost::winapi::UINT_ uFlags,
boost::winapi::SIZE_T_ uBytes);

BOOST_WINAPI_IMPORT_EXCEPT_WM boost::winapi::HLOCAL_ BOOST_WINAPI_WINAPI_CC
LocalReAlloc(
boost::winapi::HLOCAL_ hMem,
boost::winapi::SIZE_T_ uBytes,
boost::winapi::UINT_ uFlags);
#endif

BOOST_WINAPI_IMPORT_EXCEPT_WM boost::winapi::HLOCAL_ BOOST_WINAPI_WINAPI_CC LocalFree(boost::winapi::HLOCAL_ hMem);
} 
#endif

namespace boost {
namespace winapi {
#if defined( BOOST_USE_WINDOWS_H )
typedef ::HLOCAL HLOCAL_;
#endif
using ::LocalAlloc;
using ::LocalReAlloc;
using ::LocalFree;
}
}

#include <boost/winapi/detail/footer.hpp>

#endif 
#endif 
