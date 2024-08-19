

#ifndef BOOST_WINAPI_GET_CURRENT_PROCESS_ID_HPP_INCLUDED_
#define BOOST_WINAPI_GET_CURRENT_PROCESS_ID_HPP_INCLUDED_

#include <boost/winapi/basic_types.hpp>
#include <boost/winapi/detail/header.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

#if !defined( BOOST_USE_WINDOWS_H ) && !defined( UNDER_CE )
extern "C" {
BOOST_WINAPI_IMPORT boost::winapi::DWORD_ BOOST_WINAPI_WINAPI_CC GetCurrentProcessId(BOOST_WINAPI_DETAIL_VOID);
}
#endif

namespace boost {
namespace winapi {
using ::GetCurrentProcessId;
}
}

#include <boost/winapi/detail/footer.hpp>

#endif 
