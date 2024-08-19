



#ifndef BOOST_CONFIG_SUFFIX_HPP
#define BOOST_CONFIG_SUFFIX_HPP

#if defined(__GNUC__) && (__GNUC__ >= 4)
#pragma GCC system_header
#endif

#ifndef BOOST_SYMBOL_EXPORT
# define BOOST_SYMBOL_EXPORT
#endif
#ifndef BOOST_SYMBOL_IMPORT
# define BOOST_SYMBOL_IMPORT
#endif
#ifndef BOOST_SYMBOL_VISIBLE
# define BOOST_SYMBOL_VISIBLE
#endif

#if !defined(BOOST_HAS_LONG_LONG) && !defined(BOOST_NO_LONG_LONG)                                              \
&& !defined(BOOST_MSVC) && !defined(BOOST_BORLANDC)
# include <limits.h>
# if (defined(ULLONG_MAX) || defined(ULONG_LONG_MAX) || defined(ULONGLONG_MAX))
#   define BOOST_HAS_LONG_LONG
# else
#   define BOOST_NO_LONG_LONG
# endif
#endif

#if defined(__GNUC__) && (__GNUC__ >= 3) && defined(BOOST_NO_CTYPE_FUNCTIONS)
#  undef BOOST_NO_CTYPE_FUNCTIONS
#endif

#  ifndef BOOST_STD_EXTENSION_NAMESPACE
#    define BOOST_STD_EXTENSION_NAMESPACE std
#  endif

#  if defined(BOOST_NO_CV_SPECIALIZATIONS) \
&& !defined(BOOST_NO_CV_VOID_SPECIALIZATIONS)
#     define BOOST_NO_CV_VOID_SPECIALIZATIONS
#  endif

#  if defined(BOOST_NO_LIMITS) \
&& !defined(BOOST_NO_LIMITS_COMPILE_TIME_CONSTANTS)
#     define BOOST_NO_LIMITS_COMPILE_TIME_CONSTANTS
#     define BOOST_NO_MS_INT64_NUMERIC_LIMITS
#     define BOOST_NO_LONG_LONG_NUMERIC_LIMITS
#  endif

#if !defined(BOOST_HAS_LONG_LONG) && !defined(BOOST_NO_LONG_LONG_NUMERIC_LIMITS)
#  define BOOST_NO_LONG_LONG_NUMERIC_LIMITS
#endif

#if !defined(BOOST_HAS_MS_INT64) && !defined(BOOST_NO_MS_INT64_NUMERIC_LIMITS)
#  define BOOST_NO_MS_INT64_NUMERIC_LIMITS
#endif

#  if !defined(BOOST_NO_MEMBER_TEMPLATES) \
&& !defined(BOOST_MSVC6_MEMBER_TEMPLATES)
#     define BOOST_MSVC6_MEMBER_TEMPLATES
#  endif

#  if defined(BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION) \
&& !defined(BOOST_BCB_PARTIAL_SPECIALIZATION_BUG)
#     define BOOST_BCB_PARTIAL_SPECIALIZATION_BUG
#  endif

#  if defined(BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION) \
&& !defined(BOOST_NO_ARRAY_TYPE_SPECIALIZATIONS)
#     define BOOST_NO_ARRAY_TYPE_SPECIALIZATIONS
#  endif

#  if defined(BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION) \
&& !defined(BOOST_NO_STD_ITERATOR_TRAITS)
#     define BOOST_NO_STD_ITERATOR_TRAITS
#  endif

#  if defined(BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION) \
&& !defined(BOOST_NO_PARTIAL_SPECIALIZATION_IMPLICIT_DEFAULT_ARGS)
#     define BOOST_NO_PARTIAL_SPECIALIZATION_IMPLICIT_DEFAULT_ARGS
#  endif

#  if defined(BOOST_NO_MEMBER_TEMPLATES) \
&& !defined(BOOST_MSVC6_MEMBER_TEMPLATES) \
&& !defined(BOOST_NO_TEMPLATED_ITERATOR_CONSTRUCTORS)
#     define BOOST_NO_TEMPLATED_ITERATOR_CONSTRUCTORS
#  endif

#  if defined(BOOST_NO_MEMBER_TEMPLATES) \
&& !defined(BOOST_MSVC6_MEMBER_TEMPLATES) \
&& !defined(BOOST_NO_STD_ALLOCATOR)
#     define BOOST_NO_STD_ALLOCATOR
#  endif

#if defined(BOOST_NO_ARGUMENT_DEPENDENT_LOOKUP) && !defined(BOOST_FUNCTION_SCOPE_USING_DECLARATION_BREAKS_ADL)
#  define BOOST_FUNCTION_SCOPE_USING_DECLARATION_BREAKS_ADL
#endif

#if defined(BOOST_NO_TYPEID) && !defined(BOOST_NO_RTTI)
#  define BOOST_NO_RTTI
#endif

#if !defined(BOOST_NO_STD_ALLOCATOR)
#  define BOOST_HAS_PARTIAL_STD_ALLOCATOR
#endif

#  if defined(BOOST_NO_STD_LOCALE) && !defined(BOOST_NO_STD_USE_FACET)
#     define BOOST_NO_STD_USE_FACET
#  endif

#  if defined(BOOST_NO_STD_LOCALE) && !defined(BOOST_NO_STD_MESSAGES)
#     define BOOST_NO_STD_MESSAGES
#  endif

#  if defined(BOOST_NO_STD_LOCALE) && !defined(BOOST_NO_STD_WSTREAMBUF)
#     define BOOST_NO_STD_WSTREAMBUF
#  endif

#  if defined(BOOST_NO_CWCHAR) && !defined(BOOST_NO_CWCTYPE)
#     define BOOST_NO_CWCTYPE
#  endif

#  if defined(BOOST_NO_CWCHAR) && !defined(BOOST_NO_SWPRINTF)
#     define BOOST_NO_SWPRINTF
#  endif

#if defined(BOOST_DISABLE_WIN32) && defined(_WIN32) \
&& !defined(BOOST_DISABLE_THREADS) && !defined(BOOST_HAS_PTHREADS)
#  define BOOST_DISABLE_THREADS
#endif

#if (defined(__MT__) || defined(_MT) || defined(_REENTRANT) \
|| defined(_PTHREADS) || defined(__APPLE__) || defined(__DragonFly__)) \
&& !defined(BOOST_HAS_THREADS)
#  define BOOST_HAS_THREADS
#endif

#if defined(BOOST_DISABLE_THREADS) && defined(BOOST_HAS_THREADS)
#  undef BOOST_HAS_THREADS
#endif

#if defined(BOOST_HAS_THREADS) && !defined(BOOST_HAS_PTHREADS)\
&& !defined(BOOST_HAS_WINTHREADS) && !defined(BOOST_HAS_BETHREADS)\
&& !defined(BOOST_HAS_MPTASKS)
#  undef BOOST_HAS_THREADS
#endif

#ifndef BOOST_HAS_THREADS
#  undef BOOST_HAS_PTHREADS
#  undef BOOST_HAS_PTHREAD_MUTEXATTR_SETTYPE
#  undef BOOST_HAS_PTHREAD_YIELD
#  undef BOOST_HAS_PTHREAD_DELAY_NP
#  undef BOOST_HAS_WINTHREADS
#  undef BOOST_HAS_BETHREADS
#  undef BOOST_HAS_MPTASKS
#endif

#  if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901)
#     define BOOST_HAS_STDINT_H
#     ifndef BOOST_HAS_LOG1P
#        define BOOST_HAS_LOG1P
#     endif
#     ifndef BOOST_HAS_EXPM1
#        define BOOST_HAS_EXPM1
#     endif
#  endif

#  if !defined(BOOST_HAS_SLIST) && !defined(BOOST_NO_SLIST)
#     define BOOST_NO_SLIST
#  endif

#  if !defined(BOOST_HAS_HASH) && !defined(BOOST_NO_HASH)
#     define BOOST_NO_HASH
#  endif

#if defined(BOOST_HAS_SLIST) && !defined(BOOST_SLIST_HEADER)
#  define BOOST_SLIST_HEADER <slist>
#endif

#if defined(BOOST_HAS_HASH) && !defined(BOOST_HASH_SET_HEADER)
#  define BOOST_HASH_SET_HEADER <hash_set>
#endif

#if defined(BOOST_HAS_HASH) && !defined(BOOST_HASH_MAP_HEADER)
#  define BOOST_HASH_MAP_HEADER <hash_map>
#endif

#if defined(BOOST_ABI_PREFIX) && defined(BOOST_ABI_SUFFIX) && !defined(BOOST_HAS_ABI_HEADERS)
#  define BOOST_HAS_ABI_HEADERS
#endif

#if defined(BOOST_HAS_ABI_HEADERS) && defined(BOOST_DISABLE_ABI_HEADERS)
#  undef BOOST_HAS_ABI_HEADERS
#endif


# if defined(BOOST_NO_STDC_NAMESPACE) && defined(__cplusplus)
#   include <cstddef>
namespace std { using ::ptrdiff_t; using ::size_t; }
# endif


#define BOOST_PREVENT_MACRO_SUBSTITUTION

#ifndef BOOST_USING_STD_MIN
#  define BOOST_USING_STD_MIN() using std::min
#endif

#ifndef BOOST_USING_STD_MAX
#  define BOOST_USING_STD_MAX() using std::max
#endif


#  if defined(BOOST_NO_STD_MIN_MAX) && defined(__cplusplus)

namespace std {
template <class _Tp>
inline const _Tp& min BOOST_PREVENT_MACRO_SUBSTITUTION (const _Tp& __a, const _Tp& __b) {
return __b < __a ? __b : __a;
}
template <class _Tp>
inline const _Tp& max BOOST_PREVENT_MACRO_SUBSTITUTION (const _Tp& __a, const _Tp& __b) {
return  __a < __b ? __b : __a;
}
}

#  endif


#  ifdef BOOST_NO_INCLASS_MEMBER_INITIALIZATION
#       define BOOST_STATIC_CONSTANT(type, assignment) enum { assignment }
#  else
#     define BOOST_STATIC_CONSTANT(type, assignment) static const type assignment
#  endif


#if defined(BOOST_NO_STD_USE_FACET)
#  ifdef BOOST_HAS_TWO_ARG_USE_FACET
#     define BOOST_USE_FACET(Type, loc) std::use_facet(loc, static_cast<Type*>(0))
#     define BOOST_HAS_FACET(Type, loc) std::has_facet(loc, static_cast<Type*>(0))
#  elif defined(BOOST_HAS_MACRO_USE_FACET)
#     define BOOST_USE_FACET(Type, loc) std::_USE(loc, Type)
#     define BOOST_HAS_FACET(Type, loc) std::_HAS(loc, Type)
#  elif defined(BOOST_HAS_STLP_USE_FACET)
#     define BOOST_USE_FACET(Type, loc) (*std::_Use_facet<Type >(loc))
#     define BOOST_HAS_FACET(Type, loc) std::has_facet< Type >(loc)
#  endif
#else
#  define BOOST_USE_FACET(Type, loc) std::use_facet< Type >(loc)
#  define BOOST_HAS_FACET(Type, loc) std::has_facet< Type >(loc)
#endif


#ifndef BOOST_NO_MEMBER_TEMPLATE_KEYWORD
#  define BOOST_NESTED_TEMPLATE template
#else
#  define BOOST_NESTED_TEMPLATE
#endif


#ifndef BOOST_UNREACHABLE_RETURN
#  ifdef BOOST_NO_UNREACHABLE_RETURN_DETECTION
#     define BOOST_UNREACHABLE_RETURN(x) return x;
#  else
#     define BOOST_UNREACHABLE_RETURN(x)
#  endif
#endif


#ifndef BOOST_NO_DEDUCED_TYPENAME
#  define BOOST_DEDUCED_TYPENAME typename
#else
#  define BOOST_DEDUCED_TYPENAME
#endif

#ifndef BOOST_NO_TYPENAME_WITH_CTOR
#  define BOOST_CTOR_TYPENAME typename
#else
#  define BOOST_CTOR_TYPENAME
#endif

#if defined(BOOST_HAS_LONG_LONG) && defined(__cplusplus)
namespace boost{
#  ifdef __GNUC__
__extension__ typedef long long long_long_type;
__extension__ typedef unsigned long long ulong_long_type;
#  else
typedef long long long_long_type;
typedef unsigned long long ulong_long_type;
#  endif
}
#endif
#if defined(BOOST_HAS_INT128) && defined(__cplusplus)
namespace boost{
#  ifdef __GNUC__
__extension__ typedef __int128 int128_type;
__extension__ typedef unsigned __int128 uint128_type;
#  else
typedef __int128 int128_type;
typedef unsigned __int128 uint128_type;
#  endif
}
#endif
#if defined(BOOST_HAS_FLOAT128) && defined(__cplusplus)
namespace boost {
#  ifdef __GNUC__
__extension__ typedef __float128 float128_type;
#  else
typedef __float128 float128_type;
#  endif
}
#endif



#  define BOOST_EXPLICIT_TEMPLATE_TYPE(t)
#  define BOOST_EXPLICIT_TEMPLATE_TYPE_SPEC(t)
#  define BOOST_EXPLICIT_TEMPLATE_NON_TYPE(t, v)
#  define BOOST_EXPLICIT_TEMPLATE_NON_TYPE_SPEC(t, v)

#  define BOOST_APPEND_EXPLICIT_TEMPLATE_TYPE(t)
#  define BOOST_APPEND_EXPLICIT_TEMPLATE_TYPE_SPEC(t)
#  define BOOST_APPEND_EXPLICIT_TEMPLATE_NON_TYPE(t, v)
#  define BOOST_APPEND_EXPLICIT_TEMPLATE_NON_TYPE_SPEC(t, v)

#if defined(BOOST_NO_STD_TYPEINFO) && defined(__cplusplus) && defined(BOOST_MSVC)
#include <typeinfo>
namespace std{ using ::type_info; }
#undef BOOST_NO_STD_TYPEINFO
#endif



#include <boost/config/helper_macros.hpp>

#  ifndef BOOST_COMPILER
#     define BOOST_COMPILER "Unknown ISO C++ Compiler"
#  endif
#  ifndef BOOST_STDLIB
#     define BOOST_STDLIB "Unknown ISO standard library"
#  endif
#  ifndef BOOST_PLATFORM
#     if defined(unix) || defined(__unix) || defined(_XOPEN_SOURCE) \
|| defined(_POSIX_SOURCE)
#        define BOOST_PLATFORM "Generic Unix"
#     else
#        define BOOST_PLATFORM "Unknown"
#     endif
#  endif

#  ifndef BOOST_GPU_ENABLED
#  define BOOST_GPU_ENABLED
#  endif

#if !defined(BOOST_RESTRICT)
#  if defined(_MSC_VER)
#    define BOOST_RESTRICT __restrict
#    if !defined(BOOST_NO_RESTRICT_REFERENCES) && (_MSC_FULL_VER < 190023026)
#      define BOOST_NO_RESTRICT_REFERENCES
#    endif
#  elif defined(__GNUC__) && __GNUC__ > 3
#    define BOOST_RESTRICT __restrict__
#  else
#    define BOOST_RESTRICT
#    if !defined(BOOST_NO_RESTRICT_REFERENCES)
#      define BOOST_NO_RESTRICT_REFERENCES
#    endif
#  endif
#endif

#if !defined(BOOST_MAY_ALIAS)
#  define BOOST_NO_MAY_ALIAS
#  define BOOST_MAY_ALIAS
#endif

#if !defined(BOOST_FORCEINLINE)
#  if defined(_MSC_VER)
#    define BOOST_FORCEINLINE __forceinline
#  elif defined(__GNUC__) && __GNUC__ > 3
#    define BOOST_FORCEINLINE inline __attribute__ ((__always_inline__))
#  else
#    define BOOST_FORCEINLINE inline
#  endif
#endif

#if !defined(BOOST_NOINLINE)
#  if defined(_MSC_VER)
#    define BOOST_NOINLINE __declspec(noinline)
#  elif defined(__GNUC__) && __GNUC__ > 3
#    if defined(__CUDACC__)
#      define BOOST_NOINLINE __attribute__ ((noinline))
#    else
#      define BOOST_NOINLINE __attribute__ ((__noinline__))
#    endif
#  else
#    define BOOST_NOINLINE
#  endif
#endif

#if !defined(BOOST_NORETURN)
#  if defined(_MSC_VER)
#    define BOOST_NORETURN __declspec(noreturn)
#  elif defined(__GNUC__) || defined(__CODEGEARC__) && defined(__clang__)
#    define BOOST_NORETURN __attribute__ ((__noreturn__))
#  elif defined(__has_attribute) && defined(__SUNPRO_CC) && (__SUNPRO_CC > 0x5130)
#    if __has_attribute(noreturn)
#      define BOOST_NORETURN [[noreturn]]
#    endif
#  elif defined(__has_cpp_attribute) 
#    if __has_cpp_attribute(noreturn)
#      define BOOST_NORETURN [[noreturn]]
#    endif
#  endif
#endif

#if !defined(BOOST_NORETURN)
#  define BOOST_NO_NORETURN
#  define BOOST_NORETURN
#endif

#if !defined(BOOST_LIKELY)
#  define BOOST_LIKELY(x) x
#endif
#if !defined(BOOST_UNLIKELY)
#  define BOOST_UNLIKELY(x) x
#endif

#if !defined(BOOST_NO_CXX11_OVERRIDE)
#  define BOOST_OVERRIDE override
#else
#  define BOOST_OVERRIDE
#endif

#if !defined(BOOST_ALIGNMENT)
#  if !defined(BOOST_NO_CXX11_ALIGNAS)
#    define BOOST_ALIGNMENT(x) alignas(x)
#  elif defined(_MSC_VER)
#    define BOOST_ALIGNMENT(x) __declspec(align(x))
#  elif defined(__GNUC__)
#    define BOOST_ALIGNMENT(x) __attribute__ ((__aligned__(x)))
#  else
#    define BOOST_NO_ALIGNMENT
#    define BOOST_ALIGNMENT(x)
#  endif
#endif

#if !defined(BOOST_NO_CXX11_NON_PUBLIC_DEFAULTED_FUNCTIONS) && defined(BOOST_NO_CXX11_DEFAULTED_FUNCTIONS)
#  define BOOST_NO_CXX11_NON_PUBLIC_DEFAULTED_FUNCTIONS
#endif

#if !defined(BOOST_NO_CXX11_DEFAULTED_MOVES) && (defined(BOOST_NO_CXX11_DEFAULTED_FUNCTIONS) || defined(BOOST_NO_CXX11_RVALUE_REFERENCES))
#  define BOOST_NO_CXX11_DEFAULTED_MOVES
#endif

#if !(defined(BOOST_NO_CXX11_DEFAULTED_FUNCTIONS) || defined(BOOST_NO_CXX11_NON_PUBLIC_DEFAULTED_FUNCTIONS))
#   define BOOST_DEFAULTED_FUNCTION(fun, body) fun = default;
#else
#   define BOOST_DEFAULTED_FUNCTION(fun, body) fun body
#endif

#if !defined(BOOST_NO_CXX11_DELETED_FUNCTIONS)
#   define BOOST_DELETED_FUNCTION(fun) fun = delete;
#else
#   define BOOST_DELETED_FUNCTION(fun) private: fun;
#endif

#if defined(BOOST_NO_CXX11_DECLTYPE) && !defined(BOOST_NO_CXX11_DECLTYPE_N3276)
#define BOOST_NO_CXX11_DECLTYPE_N3276 BOOST_NO_CXX11_DECLTYPE
#endif


#if defined(BOOST_NO_CXX11_HDR_UNORDERED_MAP) || defined (BOOST_NO_CXX11_HDR_UNORDERED_SET)
# ifndef BOOST_NO_CXX11_STD_UNORDERED
#  define BOOST_NO_CXX11_STD_UNORDERED
# endif
#endif

#if defined(BOOST_NO_CXX11_HDR_INITIALIZER_LIST) && !defined(BOOST_NO_INITIALIZER_LISTS)
#  define BOOST_NO_INITIALIZER_LISTS
#endif

#if defined(BOOST_NO_CXX11_HDR_ARRAY) && !defined(BOOST_NO_0X_HDR_ARRAY)
#  define BOOST_NO_0X_HDR_ARRAY
#endif
#if defined(BOOST_NO_CXX11_HDR_CHRONO) && !defined(BOOST_NO_0X_HDR_CHRONO)
#  define BOOST_NO_0X_HDR_CHRONO
#endif
#if defined(BOOST_NO_CXX11_HDR_CODECVT) && !defined(BOOST_NO_0X_HDR_CODECVT)
#  define BOOST_NO_0X_HDR_CODECVT
#endif
#if defined(BOOST_NO_CXX11_HDR_CONDITION_VARIABLE) && !defined(BOOST_NO_0X_HDR_CONDITION_VARIABLE)
#  define BOOST_NO_0X_HDR_CONDITION_VARIABLE
#endif
#if defined(BOOST_NO_CXX11_HDR_FORWARD_LIST) && !defined(BOOST_NO_0X_HDR_FORWARD_LIST)
#  define BOOST_NO_0X_HDR_FORWARD_LIST
#endif
#if defined(BOOST_NO_CXX11_HDR_FUTURE) && !defined(BOOST_NO_0X_HDR_FUTURE)
#  define BOOST_NO_0X_HDR_FUTURE
#endif

#ifdef BOOST_NO_CXX11_HDR_INITIALIZER_LIST
# ifndef BOOST_NO_0X_HDR_INITIALIZER_LIST
#  define BOOST_NO_0X_HDR_INITIALIZER_LIST
# endif
# ifndef BOOST_NO_INITIALIZER_LISTS
#  define BOOST_NO_INITIALIZER_LISTS
# endif
#endif

#if defined(BOOST_NO_CXX11_HDR_MUTEX) && !defined(BOOST_NO_0X_HDR_MUTEX)
#  define BOOST_NO_0X_HDR_MUTEX
#endif
#if defined(BOOST_NO_CXX11_HDR_RANDOM) && !defined(BOOST_NO_0X_HDR_RANDOM)
#  define BOOST_NO_0X_HDR_RANDOM
#endif
#if defined(BOOST_NO_CXX11_HDR_RATIO) && !defined(BOOST_NO_0X_HDR_RATIO)
#  define BOOST_NO_0X_HDR_RATIO
#endif
#if defined(BOOST_NO_CXX11_HDR_REGEX) && !defined(BOOST_NO_0X_HDR_REGEX)
#  define BOOST_NO_0X_HDR_REGEX
#endif
#if defined(BOOST_NO_CXX11_HDR_SYSTEM_ERROR) && !defined(BOOST_NO_0X_HDR_SYSTEM_ERROR)
#  define BOOST_NO_0X_HDR_SYSTEM_ERROR
#endif
#if defined(BOOST_NO_CXX11_HDR_THREAD) && !defined(BOOST_NO_0X_HDR_THREAD)
#  define BOOST_NO_0X_HDR_THREAD
#endif
#if defined(BOOST_NO_CXX11_HDR_TUPLE) && !defined(BOOST_NO_0X_HDR_TUPLE)
#  define BOOST_NO_0X_HDR_TUPLE
#endif
#if defined(BOOST_NO_CXX11_HDR_TYPE_TRAITS) && !defined(BOOST_NO_0X_HDR_TYPE_TRAITS)
#  define BOOST_NO_0X_HDR_TYPE_TRAITS
#endif
#if defined(BOOST_NO_CXX11_HDR_TYPEINDEX) && !defined(BOOST_NO_0X_HDR_TYPEINDEX)
#  define BOOST_NO_0X_HDR_TYPEINDEX
#endif
#if defined(BOOST_NO_CXX11_HDR_UNORDERED_MAP) && !defined(BOOST_NO_0X_HDR_UNORDERED_MAP)
#  define BOOST_NO_0X_HDR_UNORDERED_MAP
#endif
#if defined(BOOST_NO_CXX11_HDR_UNORDERED_SET) && !defined(BOOST_NO_0X_HDR_UNORDERED_SET)
#  define BOOST_NO_0X_HDR_UNORDERED_SET
#endif



#if defined(BOOST_NO_CXX11_AUTO_DECLARATIONS) && !defined(BOOST_NO_AUTO_DECLARATIONS)
#  define BOOST_NO_AUTO_DECLARATIONS
#endif
#if defined(BOOST_NO_CXX11_AUTO_MULTIDECLARATIONS) && !defined(BOOST_NO_AUTO_MULTIDECLARATIONS)
#  define BOOST_NO_AUTO_MULTIDECLARATIONS
#endif
#if defined(BOOST_NO_CXX11_CHAR16_T) && !defined(BOOST_NO_CHAR16_T)
#  define BOOST_NO_CHAR16_T
#endif
#if defined(BOOST_NO_CXX11_CHAR32_T) && !defined(BOOST_NO_CHAR32_T)
#  define BOOST_NO_CHAR32_T
#endif
#if defined(BOOST_NO_CXX11_TEMPLATE_ALIASES) && !defined(BOOST_NO_TEMPLATE_ALIASES)
#  define BOOST_NO_TEMPLATE_ALIASES
#endif
#if defined(BOOST_NO_CXX11_CONSTEXPR) && !defined(BOOST_NO_CONSTEXPR)
#  define BOOST_NO_CONSTEXPR
#endif
#if defined(BOOST_NO_CXX11_DECLTYPE_N3276) && !defined(BOOST_NO_DECLTYPE_N3276)
#  define BOOST_NO_DECLTYPE_N3276
#endif
#if defined(BOOST_NO_CXX11_DECLTYPE) && !defined(BOOST_NO_DECLTYPE)
#  define BOOST_NO_DECLTYPE
#endif
#if defined(BOOST_NO_CXX11_DEFAULTED_FUNCTIONS) && !defined(BOOST_NO_DEFAULTED_FUNCTIONS)
#  define BOOST_NO_DEFAULTED_FUNCTIONS
#endif
#if defined(BOOST_NO_CXX11_DELETED_FUNCTIONS) && !defined(BOOST_NO_DELETED_FUNCTIONS)
#  define BOOST_NO_DELETED_FUNCTIONS
#endif
#if defined(BOOST_NO_CXX11_EXPLICIT_CONVERSION_OPERATORS) && !defined(BOOST_NO_EXPLICIT_CONVERSION_OPERATORS)
#  define BOOST_NO_EXPLICIT_CONVERSION_OPERATORS
#endif
#if defined(BOOST_NO_CXX11_EXTERN_TEMPLATE) && !defined(BOOST_NO_EXTERN_TEMPLATE)
#  define BOOST_NO_EXTERN_TEMPLATE
#endif
#if defined(BOOST_NO_CXX11_FUNCTION_TEMPLATE_DEFAULT_ARGS) && !defined(BOOST_NO_FUNCTION_TEMPLATE_DEFAULT_ARGS)
#  define BOOST_NO_FUNCTION_TEMPLATE_DEFAULT_ARGS
#endif
#if defined(BOOST_NO_CXX11_LAMBDAS) && !defined(BOOST_NO_LAMBDAS)
#  define BOOST_NO_LAMBDAS
#endif
#if defined(BOOST_NO_CXX11_LOCAL_CLASS_TEMPLATE_PARAMETERS) && !defined(BOOST_NO_LOCAL_CLASS_TEMPLATE_PARAMETERS)
#  define BOOST_NO_LOCAL_CLASS_TEMPLATE_PARAMETERS
#endif
#if defined(BOOST_NO_CXX11_NOEXCEPT) && !defined(BOOST_NO_NOEXCEPT)
#  define BOOST_NO_NOEXCEPT
#endif
#if defined(BOOST_NO_CXX11_NULLPTR) && !defined(BOOST_NO_NULLPTR)
#  define BOOST_NO_NULLPTR
#endif
#if defined(BOOST_NO_CXX11_RAW_LITERALS) && !defined(BOOST_NO_RAW_LITERALS)
#  define BOOST_NO_RAW_LITERALS
#endif
#if defined(BOOST_NO_CXX11_RVALUE_REFERENCES) && !defined(BOOST_NO_RVALUE_REFERENCES)
#  define BOOST_NO_RVALUE_REFERENCES
#endif
#if defined(BOOST_NO_CXX11_SCOPED_ENUMS) && !defined(BOOST_NO_SCOPED_ENUMS)
#  define BOOST_NO_SCOPED_ENUMS
#endif
#if defined(BOOST_NO_CXX11_STATIC_ASSERT) && !defined(BOOST_NO_STATIC_ASSERT)
#  define BOOST_NO_STATIC_ASSERT
#endif
#if defined(BOOST_NO_CXX11_STD_UNORDERED) && !defined(BOOST_NO_STD_UNORDERED)
#  define BOOST_NO_STD_UNORDERED
#endif
#if defined(BOOST_NO_CXX11_UNICODE_LITERALS) && !defined(BOOST_NO_UNICODE_LITERALS)
#  define BOOST_NO_UNICODE_LITERALS
#endif
#if defined(BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX) && !defined(BOOST_NO_UNIFIED_INITIALIZATION_SYNTAX)
#  define BOOST_NO_UNIFIED_INITIALIZATION_SYNTAX
#endif
#if defined(BOOST_NO_CXX11_VARIADIC_TEMPLATES) && !defined(BOOST_NO_VARIADIC_TEMPLATES)
#  define BOOST_NO_VARIADIC_TEMPLATES
#endif
#if defined(BOOST_NO_CXX11_VARIADIC_MACROS) && !defined(BOOST_NO_VARIADIC_MACROS)
#  define BOOST_NO_VARIADIC_MACROS
#endif
#if defined(BOOST_NO_CXX11_NUMERIC_LIMITS) && !defined(BOOST_NO_NUMERIC_LIMITS_LOWEST)
#  define BOOST_NO_NUMERIC_LIMITS_LOWEST
#endif


#if !defined(BOOST_NO_CXX11_FINAL)
#  define BOOST_FINAL final
#else
#  define BOOST_FINAL
#endif

#ifdef BOOST_NO_CXX11_NOEXCEPT
#  define BOOST_NOEXCEPT
#  define BOOST_NOEXCEPT_OR_NOTHROW throw()
#  define BOOST_NOEXCEPT_IF(Predicate)
#  define BOOST_NOEXCEPT_EXPR(Expression) false
#else
#  define BOOST_NOEXCEPT noexcept
#  define BOOST_NOEXCEPT_OR_NOTHROW noexcept
#  define BOOST_NOEXCEPT_IF(Predicate) noexcept((Predicate))
#  define BOOST_NOEXCEPT_EXPR(Expression) noexcept((Expression))
#endif
#ifndef BOOST_FALLTHROUGH
#  define BOOST_FALLTHROUGH ((void)0)
#endif

#if defined(BOOST_NO_CXX11_CONSTEXPR)
#define BOOST_CONSTEXPR
#define BOOST_CONSTEXPR_OR_CONST const
#else
#define BOOST_CONSTEXPR constexpr
#define BOOST_CONSTEXPR_OR_CONST constexpr
#endif
#if defined(BOOST_NO_CXX14_CONSTEXPR)
#define BOOST_CXX14_CONSTEXPR
#else
#define BOOST_CXX14_CONSTEXPR constexpr
#endif

#if !defined(BOOST_NO_CXX17_INLINE_VARIABLES)
#define BOOST_INLINE_VARIABLE inline
#else
#define BOOST_INLINE_VARIABLE
#endif
#if !defined(BOOST_NO_CXX17_IF_CONSTEXPR)
#  define BOOST_IF_CONSTEXPR if constexpr
#else
#  define BOOST_IF_CONSTEXPR if
#endif

#define BOOST_INLINE_CONSTEXPR  BOOST_INLINE_VARIABLE BOOST_CONSTEXPR_OR_CONST

#ifndef BOOST_ATTRIBUTE_UNUSED
#  define BOOST_ATTRIBUTE_UNUSED
#endif
#if defined(__has_attribute) && defined(__SUNPRO_CC) && (__SUNPRO_CC > 0x5130)
#if __has_attribute(nodiscard)
# define BOOST_ATTRIBUTE_NODISCARD [[nodiscard]]
#endif
#if __has_attribute(no_unique_address)
# define BOOST_ATTRIBUTE_NO_UNIQUE_ADDRESS [[no_unique_address]]
#endif
#elif defined(__has_cpp_attribute)
#if __has_cpp_attribute(nodiscard) && !(defined(__clang__) && (__cplusplus < 201703L))
# define BOOST_ATTRIBUTE_NODISCARD [[nodiscard]]
#endif
#if __has_cpp_attribute(no_unique_address) && !(defined(__GNUC__) && (__cplusplus < 201100))
# define BOOST_ATTRIBUTE_NO_UNIQUE_ADDRESS [[no_unique_address]]
#endif
#endif
#ifndef BOOST_ATTRIBUTE_NODISCARD
# define BOOST_ATTRIBUTE_NODISCARD
#endif
#ifndef BOOST_ATTRIBUTE_NO_UNIQUE_ADDRESS
# define BOOST_ATTRIBUTE_NO_UNIQUE_ADDRESS
#endif

#define BOOST_STATIC_CONSTEXPR  static BOOST_CONSTEXPR_OR_CONST

#if !defined(BOOST_NO_CXX11_STATIC_ASSERT) && !defined(BOOST_HAS_STATIC_ASSERT)
#  define BOOST_HAS_STATIC_ASSERT
#endif

#if !defined(BOOST_NO_CXX11_RVALUE_REFERENCES) && !defined(BOOST_HAS_RVALUE_REFS)
#define BOOST_HAS_RVALUE_REFS
#endif

#if !defined(BOOST_NO_CXX11_VARIADIC_TEMPLATES) && !defined(BOOST_HAS_VARIADIC_TMPL)
#define BOOST_HAS_VARIADIC_TMPL
#endif
#if defined(BOOST_NO_CXX11_VARIADIC_TEMPLATES) && !defined(BOOST_NO_CXX11_FIXED_LENGTH_VARIADIC_TEMPLATE_EXPANSION_PACKS)
#  define BOOST_NO_CXX11_FIXED_LENGTH_VARIADIC_TEMPLATE_EXPANSION_PACKS
#endif

#if !defined(_YVALS) && !defined(_CPPLIB_VER)  
#if (!defined(__has_include) || (__cplusplus < 201700))
#  define BOOST_NO_CXX17_HDR_OPTIONAL
#  define BOOST_NO_CXX17_HDR_STRING_VIEW
#  define BOOST_NO_CXX17_HDR_VARIANT
#else
#if !__has_include(<optional>)
#  define BOOST_NO_CXX17_HDR_OPTIONAL
#endif
#if !__has_include(<string_view>)
#  define BOOST_NO_CXX17_HDR_STRING_VIEW
#endif
#if !__has_include(<variant>)
#  define BOOST_NO_CXX17_HDR_VARIANT
#endif
#endif
#endif

#if defined(BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION) && !defined(BOOST_CONFIG_ALLOW_DEPRECATED)
#  error "You are using a compiler which lacks features which are now a minimum requirement in order to use Boost, define BOOST_CONFIG_ALLOW_DEPRECATED if you want to continue at your own risk!!!"
#endif

#endif
