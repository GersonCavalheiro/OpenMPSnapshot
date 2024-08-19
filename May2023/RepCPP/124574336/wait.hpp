

#ifndef BOOST_WINAPI_WAIT_HPP_INCLUDED_
#define BOOST_WINAPI_WAIT_HPP_INCLUDED_

#include <boost/winapi/basic_types.hpp>
#include <boost/winapi/wait_constants.hpp>
#include <boost/winapi/detail/header.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

#if !defined( BOOST_USE_WINDOWS_H )
extern "C" {

#if BOOST_WINAPI_PARTITION_APP || BOOST_WINAPI_PARTITION_SYSTEM
BOOST_WINAPI_IMPORT boost::winapi::DWORD_ BOOST_WINAPI_WINAPI_CC
WaitForSingleObjectEx(
boost::winapi::HANDLE_ hHandle,
boost::winapi::DWORD_ dwMilliseconds,
boost::winapi::BOOL_ bAlertable);
#endif

#if BOOST_WINAPI_PARTITION_DESKTOP || BOOST_WINAPI_PARTITION_SYSTEM
#if BOOST_USE_WINAPI_VERSION >= BOOST_WINAPI_VERSION_NT4
BOOST_WINAPI_IMPORT boost::winapi::DWORD_ BOOST_WINAPI_WINAPI_CC
SignalObjectAndWait(
boost::winapi::HANDLE_ hObjectToSignal,
boost::winapi::HANDLE_ hObjectToWaitOn,
boost::winapi::DWORD_ dwMilliseconds,
boost::winapi::BOOL_ bAlertable);
#endif
#endif

#if BOOST_WINAPI_PARTITION_APP_SYSTEM
BOOST_WINAPI_IMPORT_EXCEPT_WM boost::winapi::DWORD_ BOOST_WINAPI_WINAPI_CC
WaitForSingleObject(
boost::winapi::HANDLE_ hHandle,
boost::winapi::DWORD_ dwMilliseconds);

BOOST_WINAPI_IMPORT_EXCEPT_WM boost::winapi::DWORD_ BOOST_WINAPI_WINAPI_CC
WaitForMultipleObjects(
boost::winapi::DWORD_ nCount,
boost::winapi::HANDLE_ const* lpHandles,
boost::winapi::BOOL_ bWaitAll,
boost::winapi::DWORD_ dwMilliseconds);

BOOST_WINAPI_IMPORT boost::winapi::DWORD_ BOOST_WINAPI_WINAPI_CC
WaitForMultipleObjectsEx(
boost::winapi::DWORD_ nCount,
boost::winapi::HANDLE_ const* lpHandles,
boost::winapi::BOOL_ bWaitAll,
boost::winapi::DWORD_ dwMilliseconds,
boost::winapi::BOOL_ bAlertable);
#endif 

} 
#endif

namespace boost {
namespace winapi {

#if BOOST_WINAPI_PARTITION_APP || BOOST_WINAPI_PARTITION_SYSTEM
using ::WaitForSingleObjectEx;
#endif
#if BOOST_WINAPI_PARTITION_DESKTOP || BOOST_WINAPI_PARTITION_SYSTEM
#if BOOST_USE_WINAPI_VERSION >= BOOST_WINAPI_VERSION_NT4
using ::SignalObjectAndWait;
#endif
#endif

#if BOOST_WINAPI_PARTITION_APP_SYSTEM
using ::WaitForMultipleObjects;
using ::WaitForMultipleObjectsEx;
using ::WaitForSingleObject;
#endif

}
}

#include <boost/winapi/detail/footer.hpp>

#endif 
