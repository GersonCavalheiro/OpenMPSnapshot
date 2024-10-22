

#ifndef BOOST_WINAPI_FILE_MAPPING_HPP_INCLUDED_
#define BOOST_WINAPI_FILE_MAPPING_HPP_INCLUDED_

#include <boost/winapi/basic_types.hpp>
#include <boost/winapi/detail/header.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif



#if !defined( BOOST_USE_WINDOWS_H )
extern "C" {

#if BOOST_WINAPI_PARTITION_DESKTOP
#if !defined( BOOST_NO_ANSI_APIS )
BOOST_WINAPI_IMPORT boost::winapi::HANDLE_ BOOST_WINAPI_WINAPI_CC
CreateFileMappingA(
boost::winapi::HANDLE_ hFile,
::_SECURITY_ATTRIBUTES* lpFileMappingAttributes,
boost::winapi::DWORD_ flProtect,
boost::winapi::DWORD_ dwMaximumSizeHigh,
boost::winapi::DWORD_ dwMaximumSizeLow,
boost::winapi::LPCSTR_ lpName);

BOOST_WINAPI_IMPORT boost::winapi::HANDLE_ BOOST_WINAPI_WINAPI_CC
OpenFileMappingA(
boost::winapi::DWORD_ dwDesiredAccess,
boost::winapi::BOOL_ bInheritHandle,
boost::winapi::LPCSTR_ lpName);
#endif 
#endif 

#if BOOST_WINAPI_PARTITION_DESKTOP || BOOST_WINAPI_PARTITION_SYSTEM
BOOST_WINAPI_IMPORT_EXCEPT_WM boost::winapi::HANDLE_ BOOST_WINAPI_WINAPI_CC
CreateFileMappingW(
boost::winapi::HANDLE_ hFile,
::_SECURITY_ATTRIBUTES* lpFileMappingAttributes,
boost::winapi::DWORD_ flProtect,
boost::winapi::DWORD_ dwMaximumSizeHigh,
boost::winapi::DWORD_ dwMaximumSizeLow,
boost::winapi::LPCWSTR_ lpName);

BOOST_WINAPI_IMPORT_EXCEPT_WM boost::winapi::LPVOID_ BOOST_WINAPI_WINAPI_CC
MapViewOfFile(
boost::winapi::HANDLE_ hFileMappingObject,
boost::winapi::DWORD_ dwDesiredAccess,
boost::winapi::DWORD_ dwFileOffsetHigh,
boost::winapi::DWORD_ dwFileOffsetLow,
boost::winapi::SIZE_T_ dwNumberOfBytesToMap);

BOOST_WINAPI_IMPORT boost::winapi::LPVOID_ BOOST_WINAPI_WINAPI_CC
MapViewOfFileEx(
boost::winapi::HANDLE_ hFileMappingObject,
boost::winapi::DWORD_ dwDesiredAccess,
boost::winapi::DWORD_ dwFileOffsetHigh,
boost::winapi::DWORD_ dwFileOffsetLow,
boost::winapi::SIZE_T_ dwNumberOfBytesToMap,
boost::winapi::LPVOID_ lpBaseAddress);

BOOST_WINAPI_IMPORT boost::winapi::HANDLE_ BOOST_WINAPI_WINAPI_CC
OpenFileMappingW(
boost::winapi::DWORD_ dwDesiredAccess,
boost::winapi::BOOL_ bInheritHandle,
boost::winapi::LPCWSTR_ lpName);
#endif 

#if BOOST_WINAPI_PARTITION_APP || BOOST_WINAPI_PARTITION_SYSTEM
BOOST_WINAPI_IMPORT_EXCEPT_WM boost::winapi::BOOL_ BOOST_WINAPI_WINAPI_CC
FlushViewOfFile(
boost::winapi::LPCVOID_ lpBaseAddress,
boost::winapi::SIZE_T_ dwNumberOfBytesToFlush);

BOOST_WINAPI_IMPORT_EXCEPT_WM boost::winapi::BOOL_ BOOST_WINAPI_WINAPI_CC
UnmapViewOfFile(boost::winapi::LPCVOID_ lpBaseAddress);
#endif 

} 
#endif 

namespace boost {
namespace winapi {

#if defined( BOOST_USE_WINDOWS_H )

BOOST_CONSTEXPR_OR_CONST DWORD_ SEC_FILE_ = SEC_FILE;
BOOST_CONSTEXPR_OR_CONST DWORD_ SEC_IMAGE_ = SEC_IMAGE;
BOOST_CONSTEXPR_OR_CONST DWORD_ SEC_RESERVE_ = SEC_RESERVE;
BOOST_CONSTEXPR_OR_CONST DWORD_ SEC_COMMIT_ = SEC_COMMIT;
BOOST_CONSTEXPR_OR_CONST DWORD_ SEC_NOCACHE_ = SEC_NOCACHE;

BOOST_CONSTEXPR_OR_CONST DWORD_ SECTION_QUERY_ = SECTION_QUERY;
BOOST_CONSTEXPR_OR_CONST DWORD_ SECTION_MAP_WRITE_ = SECTION_MAP_WRITE;
BOOST_CONSTEXPR_OR_CONST DWORD_ SECTION_MAP_READ_ = SECTION_MAP_READ;
BOOST_CONSTEXPR_OR_CONST DWORD_ SECTION_MAP_EXECUTE_ = SECTION_MAP_EXECUTE;
BOOST_CONSTEXPR_OR_CONST DWORD_ SECTION_EXTEND_SIZE_ = SECTION_EXTEND_SIZE;
BOOST_CONSTEXPR_OR_CONST DWORD_ SECTION_ALL_ACCESS_ = SECTION_ALL_ACCESS;

BOOST_CONSTEXPR_OR_CONST DWORD_ FILE_MAP_COPY_ = FILE_MAP_COPY;
BOOST_CONSTEXPR_OR_CONST DWORD_ FILE_MAP_WRITE_ = FILE_MAP_WRITE;
BOOST_CONSTEXPR_OR_CONST DWORD_ FILE_MAP_READ_ = FILE_MAP_READ;
BOOST_CONSTEXPR_OR_CONST DWORD_ FILE_MAP_ALL_ACCESS_ = FILE_MAP_ALL_ACCESS;

#else 

BOOST_CONSTEXPR_OR_CONST DWORD_ SEC_FILE_ = 0x800000;
BOOST_CONSTEXPR_OR_CONST DWORD_ SEC_IMAGE_ = 0x1000000;
BOOST_CONSTEXPR_OR_CONST DWORD_ SEC_RESERVE_ = 0x4000000;
BOOST_CONSTEXPR_OR_CONST DWORD_ SEC_COMMIT_ = 0x8000000;
BOOST_CONSTEXPR_OR_CONST DWORD_ SEC_NOCACHE_ = 0x10000000;

BOOST_CONSTEXPR_OR_CONST DWORD_ SECTION_QUERY_ = 0x00000001;
BOOST_CONSTEXPR_OR_CONST DWORD_ SECTION_MAP_WRITE_ = 0x00000002;
BOOST_CONSTEXPR_OR_CONST DWORD_ SECTION_MAP_READ_ = 0x00000004;
BOOST_CONSTEXPR_OR_CONST DWORD_ SECTION_MAP_EXECUTE_ = 0x00000008;
BOOST_CONSTEXPR_OR_CONST DWORD_ SECTION_EXTEND_SIZE_ = 0x00000010;
BOOST_CONSTEXPR_OR_CONST DWORD_ SECTION_ALL_ACCESS_ = 0x000F001F; 

BOOST_CONSTEXPR_OR_CONST DWORD_ FILE_MAP_COPY_ = SECTION_QUERY_;
BOOST_CONSTEXPR_OR_CONST DWORD_ FILE_MAP_WRITE_ = SECTION_MAP_WRITE_;
BOOST_CONSTEXPR_OR_CONST DWORD_ FILE_MAP_READ_ = SECTION_MAP_READ_;
BOOST_CONSTEXPR_OR_CONST DWORD_ FILE_MAP_ALL_ACCESS_ = SECTION_ALL_ACCESS_;

#endif 

BOOST_CONSTEXPR_OR_CONST DWORD_ SECTION_MAP_EXECUTE_EXPLICIT_ = 0x00000020; 
BOOST_CONSTEXPR_OR_CONST DWORD_ FILE_MAP_EXECUTE_ = SECTION_MAP_EXECUTE_EXPLICIT_; 

BOOST_CONSTEXPR_OR_CONST DWORD_ SEC_PROTECTED_IMAGE_ = 0x2000000;
BOOST_CONSTEXPR_OR_CONST DWORD_ SEC_WRITECOMBINE_ = 0x40000000;
BOOST_CONSTEXPR_OR_CONST DWORD_ SEC_LARGE_PAGES_ = 0x80000000;
BOOST_CONSTEXPR_OR_CONST DWORD_ SEC_IMAGE_NO_EXECUTE_ = (SEC_IMAGE_ | SEC_NOCACHE_);

#if BOOST_WINAPI_PARTITION_DESKTOP
#if !defined( BOOST_NO_ANSI_APIS )
BOOST_FORCEINLINE HANDLE_ CreateFileMappingA(
HANDLE_ hFile,
SECURITY_ATTRIBUTES_* lpFileMappingAttributes,
DWORD_ flProtect,
DWORD_ dwMaximumSizeHigh,
DWORD_ dwMaximumSizeLow,
LPCSTR_ lpName)
{
return ::CreateFileMappingA(
hFile,
reinterpret_cast< ::_SECURITY_ATTRIBUTES* >(lpFileMappingAttributes),
flProtect,
dwMaximumSizeHigh,
dwMaximumSizeLow,
lpName);
}

BOOST_FORCEINLINE HANDLE_ create_file_mapping(
HANDLE_ hFile,
SECURITY_ATTRIBUTES_* lpFileMappingAttributes,
DWORD_ flProtect,
DWORD_ dwMaximumSizeHigh,
DWORD_ dwMaximumSizeLow,
LPCSTR_ lpName)
{
return ::CreateFileMappingA(
hFile,
reinterpret_cast< ::_SECURITY_ATTRIBUTES* >(lpFileMappingAttributes),
flProtect,
dwMaximumSizeHigh,
dwMaximumSizeLow,
lpName);
}

using ::OpenFileMappingA;

BOOST_FORCEINLINE HANDLE_ open_file_mapping(DWORD_ dwDesiredAccess, BOOL_ bInheritHandle, LPCSTR_ lpName)
{
return ::OpenFileMappingA(dwDesiredAccess, bInheritHandle, lpName);
}
#endif
#endif 

#if BOOST_WINAPI_PARTITION_DESKTOP || BOOST_WINAPI_PARTITION_SYSTEM
BOOST_FORCEINLINE HANDLE_ CreateFileMappingW(
HANDLE_ hFile,
SECURITY_ATTRIBUTES_* lpFileMappingAttributes,
DWORD_ flProtect,
DWORD_ dwMaximumSizeHigh,
DWORD_ dwMaximumSizeLow,
LPCWSTR_ lpName)
{
return ::CreateFileMappingW(
hFile,
reinterpret_cast< ::_SECURITY_ATTRIBUTES* >(lpFileMappingAttributes),
flProtect,
dwMaximumSizeHigh,
dwMaximumSizeLow,
lpName);
}

BOOST_FORCEINLINE HANDLE_ create_file_mapping(
HANDLE_ hFile,
SECURITY_ATTRIBUTES_* lpFileMappingAttributes,
DWORD_ flProtect,
DWORD_ dwMaximumSizeHigh,
DWORD_ dwMaximumSizeLow,
LPCWSTR_ lpName)
{
return ::CreateFileMappingW(
hFile,
reinterpret_cast< ::_SECURITY_ATTRIBUTES* >(lpFileMappingAttributes),
flProtect,
dwMaximumSizeHigh,
dwMaximumSizeLow,
lpName);
}

using ::MapViewOfFile;
using ::MapViewOfFileEx;
using ::OpenFileMappingW;

BOOST_FORCEINLINE HANDLE_ open_file_mapping(DWORD_ dwDesiredAccess, BOOL_ bInheritHandle, LPCWSTR_ lpName)
{
return ::OpenFileMappingW(dwDesiredAccess, bInheritHandle, lpName);
}
#endif 

#if BOOST_WINAPI_PARTITION_APP || BOOST_WINAPI_PARTITION_SYSTEM
using ::FlushViewOfFile;
using ::UnmapViewOfFile;
#endif 

}
}

#include <boost/winapi/detail/footer.hpp>

#endif 
