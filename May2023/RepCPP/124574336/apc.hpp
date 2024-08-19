

#ifndef BOOST_WINAPI_APC_HPP_INCLUDED_
#define BOOST_WINAPI_APC_HPP_INCLUDED_

#include <boost/winapi/config.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

#if BOOST_WINAPI_PARTITION_APP_SYSTEM
#if BOOST_USE_WINAPI_VERSION >= BOOST_WINAPI_VERSION_NT4
#include <boost/winapi/basic_types.hpp>

#include <boost/winapi/detail/header.hpp>

#if !defined( BOOST_USE_WINDOWS_H )
extern "C" {
typedef boost::winapi::VOID_ (BOOST_WINAPI_NTAPI_CC *PAPCFUNC)(boost::winapi::ULONG_PTR_ Parameter);

BOOST_WINAPI_IMPORT boost::winapi::DWORD_ BOOST_WINAPI_WINAPI_CC
QueueUserAPC(
PAPCFUNC pfnAPC,
boost::winapi::HANDLE_ hThread,
boost::winapi::ULONG_PTR_ dwData);
}
#endif

namespace boost {
namespace winapi {
typedef ::PAPCFUNC PAPCFUNC_;
using ::QueueUserAPC;
}
}

#include <boost/winapi/detail/footer.hpp>

#endif 
#endif 
#endif 
