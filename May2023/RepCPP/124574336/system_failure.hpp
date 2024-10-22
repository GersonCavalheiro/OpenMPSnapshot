


#ifndef BOOST_IOSTREAMS_DETAIL_SYSTEM_FAILURE_HPP_INCLUDED
#define BOOST_IOSTREAMS_DETAIL_SYSTEM_FAILURE_HPP_INCLUDED

#if defined(_MSC_VER)
# pragma once
#endif

#include <cstring>
#include <string>
#include <boost/config.hpp>
#include <boost/throw_exception.hpp>
#include <boost/iostreams/detail/config/windows_posix.hpp>
#include <boost/iostreams/detail/ios.hpp>  

#if defined(BOOST_NO_STDC_NAMESPACE) && !defined(__LIBCOMO__)
namespace std { using ::strlen; }
#endif

#ifdef BOOST_IOSTREAMS_WINDOWS
# define WIN32_LEAN_AND_MEAN  
# include <windows.h>
#else
# include <errno.h>
# include <string.h>
#endif

namespace boost { namespace iostreams { namespace detail {

inline BOOST_IOSTREAMS_FAILURE system_failure(const char* msg)
{
std::string result;
#ifdef BOOST_IOSTREAMS_WINDOWS
DWORD err;
LPVOID lpMsgBuf;
if ( (err = ::GetLastError()) != NO_ERROR &&
::FormatMessageA( FORMAT_MESSAGE_ALLOCATE_BUFFER |
FORMAT_MESSAGE_FROM_SYSTEM,
NULL,
err,
MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
(LPSTR) &lpMsgBuf,
0,
NULL ) != 0 )
{
result.reserve(std::strlen(msg) + 2 + std::strlen((LPSTR)lpMsgBuf));
result.append(msg);
result.append(": ");
result.append((LPSTR) lpMsgBuf);
::LocalFree(lpMsgBuf);
} else {
result += msg;
}
#else
const char* system_msg = errno ? strerror(errno) : "";
result.reserve(std::strlen(msg) + 2 + std::strlen(system_msg));
result.append(msg);
result.append(": ");
result.append(system_msg);
#endif
return BOOST_IOSTREAMS_FAILURE(result);
}

inline BOOST_IOSTREAMS_FAILURE system_failure(const std::string& msg)
{ return system_failure(msg.c_str()); }

inline void throw_system_failure(const char* msg)
{ boost::throw_exception(system_failure(msg)); }

inline void throw_system_failure(const std::string& msg)
{ boost::throw_exception(system_failure(msg)); }

} } } 

#endif 
