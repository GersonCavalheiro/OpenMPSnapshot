

#ifndef BOOST_WINAPI_TLS_HPP_INCLUDED_
#define BOOST_WINAPI_TLS_HPP_INCLUDED_

#include <boost/winapi/basic_types.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

#if BOOST_WINAPI_PARTITION_APP_SYSTEM

#include <boost/winapi/detail/header.hpp>

#if !defined( BOOST_USE_WINDOWS_H )
extern "C" {
#if !defined( UNDER_CE )
BOOST_WINAPI_IMPORT boost::winapi::DWORD_ BOOST_WINAPI_WINAPI_CC
TlsAlloc(BOOST_WINAPI_DETAIL_VOID);

BOOST_WINAPI_IMPORT boost::winapi::BOOL_ BOOST_WINAPI_WINAPI_CC
TlsFree(boost::winapi::DWORD_ dwTlsIndex);
#endif

BOOST_WINAPI_IMPORT_EXCEPT_WM boost::winapi::LPVOID_ BOOST_WINAPI_WINAPI_CC
TlsGetValue(boost::winapi::DWORD_ dwTlsIndex);

BOOST_WINAPI_IMPORT_EXCEPT_WM boost::winapi::BOOL_ BOOST_WINAPI_WINAPI_CC
TlsSetValue(
boost::winapi::DWORD_ dwTlsIndex,
boost::winapi::LPVOID_ lpTlsValue);
} 
#endif

namespace boost {
namespace winapi {

using ::TlsAlloc;
using ::TlsFree;
using ::TlsGetValue;
using ::TlsSetValue;

#if defined( BOOST_USE_WINDOWS_H )
BOOST_CONSTEXPR_OR_CONST DWORD_ TLS_OUT_OF_INDEXES_ = TLS_OUT_OF_INDEXES;
#else
BOOST_CONSTEXPR_OR_CONST DWORD_ TLS_OUT_OF_INDEXES_ = 0xFFFFFFFF;
#endif

BOOST_CONSTEXPR_OR_CONST DWORD_ tls_out_of_indexes = TLS_OUT_OF_INDEXES_;

}
}

#include <boost/winapi/detail/footer.hpp>

#endif 
#endif 
