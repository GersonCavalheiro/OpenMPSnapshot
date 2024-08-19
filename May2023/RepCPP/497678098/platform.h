
#ifndef SEQAN_PLATFORM_H
#define SEQAN_PLATFORM_H

#include <cinttypes>


#include <cstddef> 
#include <ciso646> 


#ifdef _MSC_VER
#define STDLIB_VS
#endif


#ifdef __GLIBCXX__
#define STDLIB_GNU
#endif


#ifdef _LIBCPP_VERSION
#define STDLIB_LLVM
#endif



#if defined(__ICC)
#define COMPILER_LINTEL
#if __ICC < 1700
#warning ICC versions older than 17.0 are not supported.
#endif
#endif


#if defined(__ICL)
#define COMPILER_WINTEL
#if __ICL < 1700
#warning Intel compiler (windows) versions older than 17.0 are not supported.
#endif
#endif


#if defined(__clang__)
#define COMPILER_CLANG
#define COMPILER_VERSION (__clang_major__ * 10000 + __clang_minor__ * 100 + __clang_patchlevel__)
#if COMPILER_VERSION < 30500
#warning Clang versions older than 3.5.0 are not supported.
#endif
#undef COMPILER_VERSION
#endif


#if defined(_MSC_VER) && !defined(COMPILER_WINTEL) && !defined(COMPILER_CLANG)
#define COMPILER_MSVC
#if _MSC_VER < 1900
#error Visual Studio versions older than version 14 / "2015" are not supported.
#endif
#endif


#if defined(__GNUC__) && !defined(COMPILER_LINTEL) && !defined(COMPILER_CLANG)
#define COMPILER_GCC
#define COMPILER_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#if COMPILER_VERSION < 40901
#warning GCC versions older than 4.9.1 are not supported.
#endif
#undef COMPILER_VERSION
#endif


#ifndef STDLIB_VS 
#if __cplusplus < 201300
#error SeqAn requires C++14! You must compile your application with -std=c++14, -std=gnu++14 or -std=c++1y.
#endif
#endif



#ifdef STDLIB_VS
#define PLATFORM_WINDOWS
#define PLATFORM_WINDOWS_VS
#else
#define PLATFORM_GCC
#endif

#if defined(PLATFORM_GCC) && defined(COMPILER_CLANG)
#define PLATFORM_CLANG
#endif

#if defined(PLATFORM_GCC) && defined(COMPILER_LINTEL)
#define PLATFORM_INTEL
#endif

#if defined(PLATFORM_GCC) && defined(COMPILER_GCC)
#define PLATFORM_GNU
#endif


#if defined(COMPILER_MSVC) || defined(COMPILER_WINTEL)
#pragma warning( disable : 4503 )
#endif




[[deprecated("Use uint64_t instead.")]]
typedef uint64_t __uint64; 
[[deprecated("Use uint32_t instead.")]]
typedef uint32_t __uint32; 
[[deprecated("Use uint16_t instead.")]]
typedef uint16_t __uint16; 
[[deprecated("Use uint8_t instead.")]]
typedef uint8_t __uint8;   

#if !(defined(COMPILER_LINTEL) || defined(STDLIB_VS))
[[deprecated("Use int64_t instead.")]]
typedef int64_t __int64;   
[[deprecated("Use int32_t instead.")]]
typedef int32_t __int32;   
[[deprecated("Use int16_t instead.")]]
typedef int16_t __int16;   
[[deprecated("Use int8_t instead.")]]
typedef int8_t __int8;     
#endif

#if defined(COMPILER_MSVC) || defined(COMPILER_WINTEL)
#define finline __forceinline
#else
#define finline __inline__
#endif

#if !defined(COMPILER_MSVC)
#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64
#endif

#ifndef _LARGEFILE_SOURCE
#define _LARGEFILE_SOURCE
#endif
#endif 



#if defined(__amd64__) || defined(__x86_64__) || defined(__aarch64__) || defined(__arch64__) || \
defined(__ia64__) || defined(__ppc64__) || defined(__PPC64__) || defined(_WIN64) || \
defined(__LP64__) || defined(_LP64)
#define SEQAN_IS_64_BIT 1
#define SEQAN_IS_32_BIT 0
#else
#define SEQAN_IS_64_BIT 0
#define SEQAN_IS_32_BIT 1
#endif



#define SEQAN_AUTO_PTR_NAME     unique_ptr
#define SEQAN_FORWARD_ARG       &&
#define SEQAN_FORWARD_CARG      &&
#define SEQAN_FORWARD_RETURN    &&
#define SEQAN_FORWARD(T, x)     std::forward<T>(x)
#define SEQAN_MOVE(x)           std::move(x)

#define SEQAN_CXX11_STL 1
#define SEQAN_CXX11_STANDARD 1
#define SEQAN_CXX11_COMPLETE 1


#define SEQAN_FUNC inline
#define SEQAN_HOST_DEVICE
#define SEQAN_HOST
#define SEQAN_DEVICE
#define SEQAN_GLOBAL

#if defined(COMPILER_GCC) || defined(COMPILER_CLANG)
#define SEQAN_RESTRICT  __restrict__
#else
#define SEQAN_RESTRICT
#endif

#if defined(COMPILER_GCC) || defined(COMPILER_CLANG) && !defined(STDLIB_VS) || defined(COMPILER_LINTEL)
#define SEQAN_LIKELY(expr) __builtin_expect(!!(expr), 1)
#define SEQAN_UNLIKELY(expr) __builtin_expect(!!(expr), 0)
#else
#define SEQAN_LIKELY(x)    (x)
#define SEQAN_UNLIKELY(x)    (x)
#endif

#if __cplusplus >= 201703L
#define SEQAN_UNUSED [[maybe_unused]]
#else
#if defined(COMPILER_GCC) || defined(COMPILER_CLANG) || defined(COMPILER_LINTEL)
#define SEQAN_UNUSED [[gnu::unused]]
#else
#define SEQAN_UNUSED
#endif 
#endif 

#define SEQAN_UNUSED_TYPEDEF SEQAN_UNUSED

#define SEQAN_FALLTHROUGH
#if defined(__has_cpp_attribute)
#if __has_cpp_attribute(fallthrough)
#undef SEQAN_FALLTHROUGH
#if __cplusplus < 201500 && defined(COMPILER_GCC)
#define SEQAN_FALLTHROUGH [[gnu::fallthrough]];
#elif __cplusplus < 201500 && defined(COMPILER_CLANG)
#define SEQAN_FALLTHROUGH [[clang::fallthrough]];
#else
#define SEQAN_FALLTHROUGH [[fallthrough]];
#endif
#endif
#endif

#ifndef SEQAN_HAS_EXECINFO
#ifdef STDLIB_VS
#define SEQAN_HAS_EXECINFO 0
#elif defined(__has_include)
#if __has_include(<execinfo.h>)
#define SEQAN_HAS_EXECINFO 1
#else
#define SEQAN_HAS_EXECINFO 0
#endif
#else 
#define SEQAN_HAS_EXECINFO 1
#endif
#endif

#ifndef SEQAN_ASYNC_IO
#if defined(__FreeBSD__)
#define SEQAN_ASYNC_IO SEQAN_IS_64_BIT
#elif defined(__has_include) && defined(__unix__)
#if __has_include(<aio.h>)
#define SEQAN_ASYNC_IO 1
#else
#define SEQAN_ASYNC_IO 0
#endif
#elif defined(__OpenBSD__)
#define SEQAN_ASYNC_IO 0
#else
#define SEQAN_ASYNC_IO 1
#endif
#endif 

#if defined(__FreeBSD__) && defined(COMPILER_CLANG)
#define COMPILER_VERSION (__clang_major__ * 10000 + __clang_minor__ * 100 + __clang_patchlevel__)
#if (COMPILER_VERSION >= 30500) && (COMPILER_VERSION < 30600)
#define SEQAN_CLANG35_FREEBSD_BUG 1
#endif
#undef COMPILER_VERSION
#endif

#ifdef __GLIBC__
#include <endian.h>
#endif 

#if defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__) || defined(__DragonFly__)
#include <sys/endian.h>
#endif 

#ifndef SEQAN_BIG_ENDIAN
#if (defined( _BYTE_ORDER  ) && ( _BYTE_ORDER   ==        _BIG_ENDIAN  )) || \
(defined(__BYTE_ORDER  ) && (__BYTE_ORDER   ==       __BIG_ENDIAN  )) || \
(defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)) || \
defined(__BIG_ENDIAN__)
#define SEQAN_BIG_ENDIAN 1
#else
#define SEQAN_BIG_ENDIAN 0
#endif
#endif 

namespace seqan
{

template <typename T>
constexpr void enforceLittleEndian(T &)
{}

#if SEQAN_BIG_ENDIAN
inline void enforceLittleEndian(int16_t & in)
{
in = htole16(in);
}
inline void enforceLittleEndian(uint16_t & in)
{
in = htole16(in);
}
inline void enforceLittleEndian(int32_t & in)
{
in = htole32(in);
}
inline void enforceLittleEndian(uint32_t & in)
{
in = htole32(in);
}
inline void enforceLittleEndian(int64_t & in)
{
in = htole64(in);
}
inline void enforceLittleEndian(uint64_t & in)
{
in = htole64(in);
}
inline void enforceLittleEndian(float & in)
{
uint32_t tmp = htole32(*reinterpret_cast<uint32_t*>(&in));
char *out = reinterpret_cast<char*>(&in);
*out = *reinterpret_cast<char*>(&tmp);
}
inline void enforceLittleEndian(double & in)
{
uint64_t tmp = htole64(*reinterpret_cast<uint64_t*>(&in));
char *out = reinterpret_cast<char*>(&in);
*out = *reinterpret_cast<char*>(&tmp);
}
#endif 

} 

#ifndef SEQAN_DEFAULT_PAGESIZE
#define SEQAN_DEFAULT_PAGESIZE 64 * 1024
#endif

#if __cplusplus >= 201703L
#define SEQAN_IF_CONSTEXPR if constexpr
#else
#define SEQAN_IF_CONSTEXPR if
#endif

#endif 
