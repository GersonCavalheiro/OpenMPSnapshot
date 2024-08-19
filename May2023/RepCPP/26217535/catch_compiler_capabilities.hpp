

#ifndef CATCH_COMPILER_CAPABILITIES_HPP_INCLUDED
#define CATCH_COMPILER_CAPABILITIES_HPP_INCLUDED



#include <catch2/internal/catch_platform.hpp>
#include <catch2/catch_user_config.hpp>

#ifdef __cplusplus

#  if (__cplusplus >= 201402L) || (defined(_MSVC_LANG) && _MSVC_LANG >= 201402L)
#    define CATCH_CPP14_OR_GREATER
#  endif

#  if (__cplusplus >= 201703L) || (defined(_MSVC_LANG) && _MSVC_LANG >= 201703L)
#    define CATCH_CPP17_OR_GREATER
#  endif

#endif

#if defined(__GNUC__) && !defined(__clang__) && !defined(__ICC) && !defined(__CUDACC__) && !defined(__LCC__)
#    define CATCH_INTERNAL_START_WARNINGS_SUPPRESSION _Pragma( "GCC diagnostic push" )
#    define CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION  _Pragma( "GCC diagnostic pop" )

#    define CATCH_INTERNAL_SUPPRESS_PARENTHESES_WARNINGS \
_Pragma( "GCC diagnostic ignored \"-Wparentheses\"" )

#    define CATCH_INTERNAL_SUPPRESS_UNUSED_VARIABLE_WARNINGS \
_Pragma( "GCC diagnostic ignored \"-Wunused-variable\"" )

#    define CATCH_INTERNAL_SUPPRESS_USELESS_CAST_WARNINGS \
_Pragma( "GCC diagnostic ignored \"-Wuseless-cast\"" )

#    define CATCH_INTERNAL_IGNORE_BUT_WARN(...) (void)__builtin_constant_p(__VA_ARGS__)

#endif

#if defined(__CUDACC__) && !defined(__clang__)
#    define CATCH_INTERNAL_START_WARNINGS_SUPPRESSION _Pragma( "nv_diagnostic push" )
#    define CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION  _Pragma( "nv_diagnostic pop" )
#    define CATCH_INTERNAL_SUPPRESS_UNUSED_VARIABLE_WARNINGS _Pragma( "nv_diag_suppress 177" )
#endif

#if defined(__clang__) && !defined(_MSC_VER)

#    define CATCH_INTERNAL_START_WARNINGS_SUPPRESSION _Pragma( "clang diagnostic push" )
#    define CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION  _Pragma( "clang diagnostic pop" )

#endif 

#if defined(__clang__)

#  if !defined(__ibmxl__) && !defined(__CUDACC__) && !defined( __NVCOMPILER )
#    define CATCH_INTERNAL_IGNORE_BUT_WARN(...) (void)__builtin_constant_p(__VA_ARGS__) 
#  endif


#    define CATCH_INTERNAL_SUPPRESS_GLOBALS_WARNINGS \
_Pragma( "clang diagnostic ignored \"-Wexit-time-destructors\"" ) \
_Pragma( "clang diagnostic ignored \"-Wglobal-constructors\"")

#    define CATCH_INTERNAL_SUPPRESS_PARENTHESES_WARNINGS \
_Pragma( "clang diagnostic ignored \"-Wparentheses\"" )

#    define CATCH_INTERNAL_SUPPRESS_UNUSED_VARIABLE_WARNINGS \
_Pragma( "clang diagnostic ignored \"-Wunused-variable\"" )

#    define CATCH_INTERNAL_SUPPRESS_ZERO_VARIADIC_WARNINGS \
_Pragma( "clang diagnostic ignored \"-Wgnu-zero-variadic-macro-arguments\"" )

#    define CATCH_INTERNAL_SUPPRESS_UNUSED_TEMPLATE_WARNINGS \
_Pragma( "clang diagnostic ignored \"-Wunused-template\"" )

#    define CATCH_INTERNAL_SUPPRESS_COMMA_WARNINGS \
_Pragma( "clang diagnostic ignored \"-Wcomma\"" )

#endif 


#if defined( CATCH_PLATFORM_WINDOWS ) ||                                       \
defined( CATCH_PLATFORM_PLAYSTATION ) ||                                   \
defined( __CYGWIN__ ) ||                                                   \
defined( __QNX__ ) ||                                                      \
defined( __EMSCRIPTEN__ ) ||                                               \
defined( __DJGPP__ ) ||                                                    \
defined( __OS400__ )
#    define CATCH_INTERNAL_CONFIG_NO_POSIX_SIGNALS
#else
#    define CATCH_INTERNAL_CONFIG_POSIX_SIGNALS
#endif

#if defined(CATCH_PLATFORM_WINDOWS_UWP) || defined(CATCH_PLATFORM_PLAYSTATION)
#    define CATCH_INTERNAL_CONFIG_NO_GETENV
#else
#    define CATCH_INTERNAL_CONFIG_GETENV
#endif

#if defined(__ANDROID__)
#    define CATCH_INTERNAL_CONFIG_NO_CPP11_TO_STRING
#endif

#if defined(__MINGW32__)
#    define CATCH_INTERNAL_CONFIG_NO_WINDOWS_SEH
#endif

#if defined(__ORBIS__)
#    define CATCH_INTERNAL_CONFIG_NO_NEW_CAPTURE
#endif

#ifdef __CYGWIN__

#   define _BSD_SOURCE
# if !((__cplusplus >= 201103L) && defined(_GLIBCXX_USE_C99) \
&& !defined(_GLIBCXX_HAVE_BROKEN_VSWPRINTF))

#    define CATCH_INTERNAL_CONFIG_NO_CPP11_TO_STRING

# endif
#endif 

#if defined(_MSC_VER)

#  define CATCH_INTERNAL_START_WARNINGS_SUPPRESSION __pragma( warning(push) )
#  define CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION  __pragma( warning(pop) )

#  if defined(CATCH_PLATFORM_WINDOWS_UWP)
#    define CATCH_INTERNAL_CONFIG_NO_COLOUR_WIN32
#  else
#    define CATCH_INTERNAL_CONFIG_WINDOWS_SEH
#  endif

#  if !defined(__clang__) 
#    if !defined(_MSVC_TRADITIONAL) || (defined(_MSVC_TRADITIONAL) && _MSVC_TRADITIONAL)
#      define CATCH_INTERNAL_CONFIG_TRADITIONAL_MSVC_PREPROCESSOR
#    endif 
#  endif 

#endif 

#if defined(_REENTRANT) || defined(_MSC_VER)
# define CATCH_INTERNAL_CONFIG_USE_ASYNC
#endif 

#if defined(__EXCEPTIONS) || defined(__cpp_exceptions) || defined(_CPPUNWIND)
#  define CATCH_INTERNAL_CONFIG_EXCEPTIONS_ENABLED
#endif


#if defined(__BORLANDC__)
#define CATCH_INTERNAL_CONFIG_POLYFILL_ISNAN
#endif


#if defined(UNDER_RTSS) || defined(RTX64_BUILD)
#define CATCH_INTERNAL_CONFIG_NO_WINDOWS_SEH
#define CATCH_INTERNAL_CONFIG_NO_ASYNC
#define CATCH_INTERNAL_CONFIG_NO_COLOUR_WIN32
#endif

#if !defined(_GLIBCXX_USE_C99_MATH_TR1)
#define CATCH_INTERNAL_CONFIG_GLOBAL_NEXTAFTER
#endif

#if defined(__has_include)
#if __has_include(<string_view>) && defined(CATCH_CPP17_OR_GREATER)
#    define CATCH_INTERNAL_CONFIG_CPP17_STRING_VIEW
#endif

#  if __has_include(<optional>) && defined(CATCH_CPP17_OR_GREATER)
#    define CATCH_INTERNAL_CONFIG_CPP17_OPTIONAL
#  endif 

#  if __has_include(<cstddef>) && defined(CATCH_CPP17_OR_GREATER)
#    include <cstddef>
#    if defined(__cpp_lib_byte) && (__cpp_lib_byte > 0)
#      define CATCH_INTERNAL_CONFIG_CPP17_BYTE
#    endif
#  endif 

#  if __has_include(<variant>) && defined(CATCH_CPP17_OR_GREATER)
#    if defined(__clang__) && (__clang_major__ < 8)
#      include <ciso646>
#      if defined(__GLIBCXX__) && defined(_GLIBCXX_RELEASE) && (_GLIBCXX_RELEASE < 9)
#        define CATCH_CONFIG_NO_CPP17_VARIANT
#      else
#        define CATCH_INTERNAL_CONFIG_CPP17_VARIANT
#      endif 
#    else
#      define CATCH_INTERNAL_CONFIG_CPP17_VARIANT
#    endif 
#  endif 
#endif 


#if defined(CATCH_INTERNAL_CONFIG_WINDOWS_SEH) && !defined(CATCH_CONFIG_NO_WINDOWS_SEH) && !defined(CATCH_CONFIG_WINDOWS_SEH) && !defined(CATCH_INTERNAL_CONFIG_NO_WINDOWS_SEH)
#   define CATCH_CONFIG_WINDOWS_SEH
#endif
#if defined(CATCH_INTERNAL_CONFIG_POSIX_SIGNALS) && !defined(CATCH_INTERNAL_CONFIG_NO_POSIX_SIGNALS) && !defined(CATCH_CONFIG_NO_POSIX_SIGNALS) && !defined(CATCH_CONFIG_POSIX_SIGNALS)
#   define CATCH_CONFIG_POSIX_SIGNALS
#endif

#if defined(CATCH_INTERNAL_CONFIG_GETENV) && !defined(CATCH_INTERNAL_CONFIG_NO_GETENV) && !defined(CATCH_CONFIG_NO_GETENV) && !defined(CATCH_CONFIG_GETENV)
#   define CATCH_CONFIG_GETENV
#endif

#if !defined(CATCH_INTERNAL_CONFIG_NO_CPP11_TO_STRING) && !defined(CATCH_CONFIG_NO_CPP11_TO_STRING) && !defined(CATCH_CONFIG_CPP11_TO_STRING)
#    define CATCH_CONFIG_CPP11_TO_STRING
#endif

#if defined(CATCH_INTERNAL_CONFIG_CPP17_OPTIONAL) && !defined(CATCH_CONFIG_NO_CPP17_OPTIONAL) && !defined(CATCH_CONFIG_CPP17_OPTIONAL)
#  define CATCH_CONFIG_CPP17_OPTIONAL
#endif

#if defined(CATCH_INTERNAL_CONFIG_CPP17_STRING_VIEW) && !defined(CATCH_CONFIG_NO_CPP17_STRING_VIEW) && !defined(CATCH_CONFIG_CPP17_STRING_VIEW)
#  define CATCH_CONFIG_CPP17_STRING_VIEW
#endif

#if defined(CATCH_INTERNAL_CONFIG_CPP17_VARIANT) && !defined(CATCH_CONFIG_NO_CPP17_VARIANT) && !defined(CATCH_CONFIG_CPP17_VARIANT)
#  define CATCH_CONFIG_CPP17_VARIANT
#endif

#if defined(CATCH_INTERNAL_CONFIG_CPP17_BYTE) && !defined(CATCH_CONFIG_NO_CPP17_BYTE) && !defined(CATCH_CONFIG_CPP17_BYTE)
#  define CATCH_CONFIG_CPP17_BYTE
#endif


#if defined(CATCH_CONFIG_EXPERIMENTAL_REDIRECT)
#  define CATCH_INTERNAL_CONFIG_NEW_CAPTURE
#endif

#if defined(CATCH_INTERNAL_CONFIG_NEW_CAPTURE) && !defined(CATCH_INTERNAL_CONFIG_NO_NEW_CAPTURE) && !defined(CATCH_CONFIG_NO_NEW_CAPTURE) && !defined(CATCH_CONFIG_NEW_CAPTURE)
#  define CATCH_CONFIG_NEW_CAPTURE
#endif

#if !defined( CATCH_INTERNAL_CONFIG_EXCEPTIONS_ENABLED ) && \
!defined( CATCH_CONFIG_DISABLE_EXCEPTIONS ) &&          \
!defined( CATCH_CONFIG_NO_DISABLE_EXCEPTIONS )
#  define CATCH_CONFIG_DISABLE_EXCEPTIONS
#endif

#if defined(CATCH_INTERNAL_CONFIG_POLYFILL_ISNAN) && !defined(CATCH_CONFIG_NO_POLYFILL_ISNAN) && !defined(CATCH_CONFIG_POLYFILL_ISNAN)
#  define CATCH_CONFIG_POLYFILL_ISNAN
#endif

#if defined(CATCH_INTERNAL_CONFIG_USE_ASYNC)  && !defined(CATCH_INTERNAL_CONFIG_NO_ASYNC) && !defined(CATCH_CONFIG_NO_USE_ASYNC) && !defined(CATCH_CONFIG_USE_ASYNC)
#  define CATCH_CONFIG_USE_ASYNC
#endif

#if defined(CATCH_INTERNAL_CONFIG_GLOBAL_NEXTAFTER) && !defined(CATCH_CONFIG_NO_GLOBAL_NEXTAFTER) && !defined(CATCH_CONFIG_GLOBAL_NEXTAFTER)
#  define CATCH_CONFIG_GLOBAL_NEXTAFTER
#endif


#if !defined(CATCH_INTERNAL_START_WARNINGS_SUPPRESSION)
#   define CATCH_INTERNAL_START_WARNINGS_SUPPRESSION
#endif
#if !defined(CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION)
#   define CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION
#endif
#if !defined(CATCH_INTERNAL_SUPPRESS_PARENTHESES_WARNINGS)
#   define CATCH_INTERNAL_SUPPRESS_PARENTHESES_WARNINGS
#endif
#if !defined(CATCH_INTERNAL_SUPPRESS_GLOBALS_WARNINGS)
#   define CATCH_INTERNAL_SUPPRESS_GLOBALS_WARNINGS
#endif
#if !defined(CATCH_INTERNAL_SUPPRESS_UNUSED_VARIABLE_WARNINGS)
#   define CATCH_INTERNAL_SUPPRESS_UNUSED_VARIABLE_WARNINGS
#endif
#if !defined(CATCH_INTERNAL_SUPPRESS_USELESS_CAST_WARNINGS)
#   define CATCH_INTERNAL_SUPPRESS_USELESS_CAST_WARNINGS
#endif
#if !defined(CATCH_INTERNAL_SUPPRESS_ZERO_VARIADIC_WARNINGS)
#   define CATCH_INTERNAL_SUPPRESS_ZERO_VARIADIC_WARNINGS
#endif

#if !defined(CATCH_INTERNAL_IGNORE_BUT_WARN)
#   define CATCH_INTERNAL_IGNORE_BUT_WARN(...)
#endif

#if defined(__APPLE__) && defined(__apple_build_version__) && (__clang_major__ < 10)
#   undef CATCH_INTERNAL_SUPPRESS_UNUSED_TEMPLATE_WARNINGS
#elif defined(__clang__) && (__clang_major__ < 5)
#   undef CATCH_INTERNAL_SUPPRESS_UNUSED_TEMPLATE_WARNINGS
#endif

#if !defined(CATCH_INTERNAL_SUPPRESS_UNUSED_TEMPLATE_WARNINGS)
#   define CATCH_INTERNAL_SUPPRESS_UNUSED_TEMPLATE_WARNINGS
#endif

#if !defined(CATCH_INTERNAL_SUPPRESS_COMMA_WARNINGS)
#   define CATCH_INTERNAL_SUPPRESS_COMMA_WARNINGS
#endif

#if defined(CATCH_CONFIG_DISABLE_EXCEPTIONS)
#define CATCH_TRY if ((true))
#define CATCH_CATCH_ALL if ((false))
#define CATCH_CATCH_ANON(type) if ((false))
#else
#define CATCH_TRY try
#define CATCH_CATCH_ALL catch (...)
#define CATCH_CATCH_ANON(type) catch (type)
#endif

#if defined(CATCH_INTERNAL_CONFIG_TRADITIONAL_MSVC_PREPROCESSOR) && !defined(CATCH_CONFIG_NO_TRADITIONAL_MSVC_PREPROCESSOR) && !defined(CATCH_CONFIG_TRADITIONAL_MSVC_PREPROCESSOR)
#define CATCH_CONFIG_TRADITIONAL_MSVC_PREPROCESSOR
#endif

#if defined( CATCH_PLATFORM_WINDOWS ) &&       \
!defined( CATCH_CONFIG_COLOUR_WIN32 ) && \
!defined( CATCH_CONFIG_NO_COLOUR_WIN32 ) && \
!defined( CATCH_INTERNAL_CONFIG_NO_COLOUR_WIN32 )
#    define CATCH_CONFIG_COLOUR_WIN32
#endif

#if defined( CATCH_CONFIG_SHARED_LIBRARY ) && defined( _MSC_VER ) && \
!defined( CATCH_CONFIG_STATIC )
#    ifdef Catch2_EXPORTS
#        define CATCH_EXPORT 
#    else
#        define CATCH_EXPORT __declspec( dllimport )
#    endif
#else
#    define CATCH_EXPORT
#endif

#endif 
