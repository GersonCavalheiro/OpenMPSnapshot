

#ifndef BOOST_WINAPI_BCRYPT_HPP_INCLUDED_
#define BOOST_WINAPI_BCRYPT_HPP_INCLUDED_

#include <boost/winapi/basic_types.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

#if BOOST_USE_WINAPI_VERSION >= BOOST_WINAPI_VERSION_WIN6

#if BOOST_WINAPI_PARTITION_APP_SYSTEM

#if defined(BOOST_USE_WINDOWS_H)
#include <bcrypt.h>
#endif

#include <boost/winapi/detail/header.hpp>

#if defined(BOOST_USE_WINDOWS_H)

namespace boost { namespace winapi {
typedef ::BCRYPT_ALG_HANDLE BCRYPT_ALG_HANDLE_;
}}

#else 

namespace boost { namespace winapi {
typedef PVOID_ BCRYPT_ALG_HANDLE_;
}}

extern "C" {

boost::winapi::NTSTATUS_ BOOST_WINAPI_WINAPI_CC
BCryptCloseAlgorithmProvider(
boost::winapi::BCRYPT_ALG_HANDLE_ hAlgorithm,
boost::winapi::ULONG_             dwFlags
);

boost::winapi::NTSTATUS_ BOOST_WINAPI_WINAPI_CC
BCryptGenRandom(
boost::winapi::BCRYPT_ALG_HANDLE_ hAlgorithm,
boost::winapi::PUCHAR_            pbBuffer,
boost::winapi::ULONG_             cbBuffer,
boost::winapi::ULONG_             dwFlags
);

boost::winapi::NTSTATUS_ BOOST_WINAPI_WINAPI_CC
BCryptOpenAlgorithmProvider(
boost::winapi::BCRYPT_ALG_HANDLE_ *phAlgorithm,
boost::winapi::LPCWSTR_           pszAlgId,
boost::winapi::LPCWSTR_           pszImplementation,
boost::winapi::DWORD_             dwFlags
);

} 

#endif 

namespace boost {
namespace winapi {

#if defined(BOOST_USE_WINDOWS_H)
const WCHAR_ BCRYPT_RNG_ALGORITHM_[] = BCRYPT_RNG_ALGORITHM;
#else
const WCHAR_ BCRYPT_RNG_ALGORITHM_[] = L"RNG";
#endif

using ::BCryptCloseAlgorithmProvider;
using ::BCryptGenRandom;
using ::BCryptOpenAlgorithmProvider;

} 
} 

#include <boost/winapi/detail/footer.hpp>

#endif 

#endif 

#endif 
