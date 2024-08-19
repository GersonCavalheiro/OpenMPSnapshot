
#pragma once

#define _CRT_SECURE_NO_WARNINGS

#include <cstddef>
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <cstring>
#include <stdint.h>
#include <functional>



#if defined(__x86_64__) || defined(__ia64__) || defined(_M_X64)
#define __X86_64__
#endif


#if defined(linux) || defined(__linux__) || defined(__LINUX__)
#  if !defined(__LINUX__)
#     define __LINUX__
#  endif
#  if !defined(__UNIX__)
#     define __UNIX__
#  endif
#endif


#if defined(__FreeBSD__) || defined(__FREEBSD__)
#  if !defined(__FREEBSD__)
#     define __FREEBSD__
#  endif
#  if !defined(__UNIX__)
#     define __UNIX__
#  endif
#endif


#if (defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)) && !defined(__CYGWIN__)
#  if !defined(__WIN32__)
#     define __WIN32__
#  endif
#endif


#if defined(__CYGWIN__)
#  if !defined(__UNIX__)
#     define __UNIX__
#  endif
#endif


#if defined(__APPLE__) || defined(MACOSX) || defined(__MACOSX__)
#  if !defined(__MACOSX__)
#     define __MACOSX__
#  endif
#  if !defined(__UNIX__)
#     define __UNIX__
#  endif
#endif


#if defined(__unix__) || defined (unix) || defined(__unix) || defined(_unix)
#  if !defined(__UNIX__)
#     define __UNIX__
#  endif
#endif

#if defined (_DEBUG)
#define DEBUG
#endif


#ifdef __WIN32__
#define __dllexport __declspec(dllexport)
#define __dllimport __declspec(dllimport)
#else
#define __dllexport __attribute__ ((visibility ("default")))
#define __dllimport
#endif

#ifdef __WIN32__
#undef __noinline
#define __noinline             __declspec(noinline)
#if defined(__INTEL_COMPILER)
#define __restrict__           __restrict
#else
#define __restrict__           
#endif
#define __thread               __declspec(thread)
#define __aligned(...)           __declspec(align(__VA_ARGS__))
#define debugbreak()           __debugbreak()

#else
#if !defined(__noinline)
#define __noinline             __attribute__((noinline))
#endif
#if !defined(__forceinline)
#define __forceinline          inline __attribute__((always_inline))
#endif
#if !defined(__aligned)
#define __aligned(...)           __attribute__((aligned(__VA_ARGS__)))
#endif
#if !defined(__FUNCTION__)
#define __FUNCTION__           __PRETTY_FUNCTION__
#endif
#define debugbreak()           asm ("int $3")
#endif

#if defined(__clang__) || defined(__GNUC__)
#define MAYBE_UNUSED __attribute__((unused))
#else
#define MAYBE_UNUSED
#endif

#if defined(_MSC_VER) && (_MSC_VER < 1900) 
#define DELETED
#else
#define DELETED  = delete
#endif

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
#define   likely(expr) (expr)
#define unlikely(expr) (expr)
#else
#define   likely(expr) __builtin_expect((bool)(expr),true )
#define unlikely(expr) __builtin_expect((bool)(expr),false)
#endif


#if 0

#define STRING(x) #x
#define TOSTRING(x) STRING(x)
#define PING std::cout << __FILE__ << " (" << __LINE__ << "): " << __FUNCTION__ << std::endl
#define PRINT(x) std::cout << STRING(x) << " = " << (x) << std::endl
#define PRINT2(x,y) std::cout << STRING(x) << " = " << (x) << ", " << STRING(y) << " = " << (y) << std::endl
#define PRINT3(x,y,z) std::cout << STRING(x) << " = " << (x) << ", " << STRING(y) << " = " << (y) << ", " << STRING(z) << " = " << (z) << std::endl
#define PRINT4(x,y,z,w) std::cout << STRING(x) << " = " << (x) << ", " << STRING(y) << " = " << (y) << ", " << STRING(z) << " = " << (z) << ", " << STRING(w) << " = " << (w) << std::endl

#if defined(DEBUG) 
#define THROW_RUNTIME_ERROR(str) \
throw std::runtime_error(std::string(__FILE__) + " (" + toString(__LINE__) + "): " + std::string(str));
#else
#define THROW_RUNTIME_ERROR(str) \
throw std::runtime_error(str);
#endif

#define FATAL(x)   THROW_RUNTIME_ERROR(x)
#define WARNING(x) { std::cerr << "Warning: " << x << std::endl << std::flush; }

#define NOT_IMPLEMENTED FATAL(std::string(__FUNCTION__) + " not implemented")
#endif



typedef float real;


#if defined(__WIN32__)
#if defined(__X86_64__)
typedef int64_t ssize_t;
#else
typedef int32_t ssize_t;
#endif
#endif


__forceinline std::string toString(long long value) {
return std::to_string(value);
}


#if defined(__INTEL_COMPILER)
#pragma warning(disable:2196) 
#pragma warning(disable:15335)  
#endif

#if defined(_MSC_VER)
#pragma warning(disable:4800) 
#pragma warning(disable:4503) 
#pragma warning(disable:4180) 
#pragma warning(disable:4258) 
#pragma warning(disable:4789) 
#endif

#if defined(__clang__) && !defined(__INTEL_COMPILER)
#endif

#if defined(__GNUC__) && !defined(__INTEL_COMPILER) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#pragma GCC diagnostic ignored "-Wattributes"
#pragma GCC diagnostic ignored "-Wmisleading-indentation"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wparentheses"
#endif

#if defined(__clang__) && defined(__WIN32__)
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wmicrosoft-cast"
#pragma clang diagnostic ignored "-Wmicrosoft-enum-value"
#pragma clang diagnostic ignored "-Wmicrosoft-include"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunknown-pragmas"
#endif


#if defined(__WIN32__) && defined(__INTEL_COMPILER)
#define DISABLE_DEPRECATED_WARNING __pragma(warning (disable: 1478)) 
#define ENABLE_DEPRECATED_WARNING  __pragma(warning (enable:  1478)) 
#elif defined(__INTEL_COMPILER)
#define DISABLE_DEPRECATED_WARNING _Pragma("warning (disable: 1478)") 
#define ENABLE_DEPRECATED_WARNING  _Pragma("warning (enable : 1478)") 
#elif defined(__clang__)
#define DISABLE_DEPRECATED_WARNING _Pragma("clang diagnostic ignored \"-Wdeprecated-declarations\"") 
#define ENABLE_DEPRECATED_WARNING  _Pragma("clang diagnostic warning \"-Wdeprecated-declarations\"") 
#elif defined(__GNUC__)
#define DISABLE_DEPRECATED_WARNING _Pragma("GCC diagnostic ignored \"-Wdeprecated-declarations\"") 
#define ENABLE_DEPRECATED_WARNING  _Pragma("GCC diagnostic warning \"-Wdeprecated-declarations\"") 
#elif defined(_MSC_VER)
#define DISABLE_DEPRECATED_WARNING __pragma(warning (disable: 4996)) 
#define ENABLE_DEPRECATED_WARNING  __pragma(warning (enable : 4996)) 
#endif


#if defined (__GNUC__)
#define IACA_SSC_MARK( MARK_ID )						\
__asm__ __volatile__ (									\
"\n\t  movl $"#MARK_ID", %%ebx"	\
"\n\t  .byte 0x64, 0x67, 0x90"	\
: : : "memory" );

#define IACA_UD_BYTES __asm__ __volatile__ ("\n\t .byte 0x0F, 0x0B");

#else
#define IACA_UD_BYTES {__asm _emit 0x0F \
__asm _emit 0x0B}

#define IACA_SSC_MARK(x) {__asm  mov ebx, x\
__asm  _emit 0x64 \
__asm  _emit 0x67 \
__asm  _emit 0x90 }

#define IACA_VC64_START __writegsbyte(111, 111);
#define IACA_VC64_END   __writegsbyte(222, 222);

#endif

#define IACA_START {IACA_UD_BYTES \
IACA_SSC_MARK(111)}
#define IACA_END {IACA_SSC_MARK(222) \
IACA_UD_BYTES}

namespace embree
{
template<typename Closure>
struct OnScopeExitHelper
{
OnScopeExitHelper (const Closure f) : active(true), f(f) {}
~OnScopeExitHelper() { if (active) f(); }
void deactivate() { active = false; }
bool active;
const Closure f;
};

template <typename Closure>
OnScopeExitHelper<Closure> OnScopeExit(const Closure f) {
return OnScopeExitHelper<Closure>(f);
}

#define STRING_JOIN2(arg1, arg2) DO_STRING_JOIN2(arg1, arg2)
#define DO_STRING_JOIN2(arg1, arg2) arg1 ## arg2
#define ON_SCOPE_EXIT(code)                                             \
auto STRING_JOIN2(on_scope_exit_, __LINE__) = OnScopeExit([&](){code;})

template<typename Ty>
std::unique_ptr<Ty> make_unique(Ty* ptr) {
return std::unique_ptr<Ty>(ptr);
}

}
