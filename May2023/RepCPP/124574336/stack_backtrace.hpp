

#ifndef BOOST_WINAPI_STACK_BACKTRACE_HPP_INCLUDED_
#define BOOST_WINAPI_STACK_BACKTRACE_HPP_INCLUDED_

#include <boost/winapi/basic_types.hpp>
#include <boost/winapi/config.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

#if !defined( BOOST_WINAPI_IS_MINGW )

#if (BOOST_USE_NTDDI_VERSION > BOOST_WINAPI_NTDDI_WINXP)

#if BOOST_WINAPI_PARTITION_APP_SYSTEM

#include <boost/winapi/detail/header.hpp>

#if !defined( BOOST_USE_WINDOWS_H ) || (defined(_MSC_VER) && (_MSC_VER+0) < 1500)
extern "C" {

BOOST_WINAPI_IMPORT boost::winapi::WORD_
BOOST_WINAPI_NTAPI_CC RtlCaptureStackBackTrace(
boost::winapi::DWORD_ FramesToSkip,
boost::winapi::DWORD_ FramesToCapture,
boost::winapi::PVOID_* BackTrace,
boost::winapi::PDWORD_ BackTraceHash);

} 
#endif

namespace boost {
namespace winapi {

using ::RtlCaptureStackBackTrace;

}
}

#include <boost/winapi/detail/footer.hpp>

#endif 
#endif 
#endif 
#endif 
