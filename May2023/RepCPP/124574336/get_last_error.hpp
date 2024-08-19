

#ifndef BOOST_WINAPI_GET_LAST_ERROR_HPP_INCLUDED_
#define BOOST_WINAPI_GET_LAST_ERROR_HPP_INCLUDED_

#include <boost/winapi/basic_types.hpp>
#include <boost/winapi/detail/header.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

#if !defined( BOOST_USE_WINDOWS_H )
extern "C" {
BOOST_WINAPI_IMPORT_EXCEPT_WM boost::winapi::DWORD_ BOOST_WINAPI_WINAPI_CC GetLastError(BOOST_WINAPI_DETAIL_VOID);
}
#endif

namespace boost {
namespace winapi {
using ::GetLastError;
}
}

#include <boost/winapi/detail/footer.hpp>

#endif 
