

#ifndef BOOST_WINAPI_THREAD_HPP_INCLUDED_
#define BOOST_WINAPI_THREAD_HPP_INCLUDED_

#include <boost/winapi/basic_types.hpp>
#include <boost/winapi/get_current_thread.hpp>
#include <boost/winapi/get_current_thread_id.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

#if BOOST_WINAPI_PARTITION_APP_SYSTEM

#include <boost/winapi/detail/header.hpp>

#if !defined( BOOST_USE_WINDOWS_H )
extern "C" {
BOOST_WINAPI_IMPORT boost::winapi::DWORD_ BOOST_WINAPI_WINAPI_CC
SleepEx(
boost::winapi::DWORD_ dwMilliseconds,
boost::winapi::BOOL_ bAlertable);
BOOST_WINAPI_IMPORT_EXCEPT_WM boost::winapi::VOID_ BOOST_WINAPI_WINAPI_CC Sleep(boost::winapi::DWORD_ dwMilliseconds);
BOOST_WINAPI_IMPORT boost::winapi::BOOL_ BOOST_WINAPI_WINAPI_CC SwitchToThread(BOOST_WINAPI_DETAIL_VOID);
} 
#endif

namespace boost {
namespace winapi {
using ::SleepEx;
using ::Sleep;
using ::SwitchToThread;
}
}

#include <boost/winapi/detail/footer.hpp>

#endif 
#endif 
