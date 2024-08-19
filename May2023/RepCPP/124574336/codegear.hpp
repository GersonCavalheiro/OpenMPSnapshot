


#if (__CODEGEARC__ > 0x740)
#  if defined(BOOST_ASSERT_CONFIG)
#     error "boost: Unknown compiler version - please run the configure tests and report the results"
#  else
#     pragma message( "boost: Unknown compiler version - please run the configure tests and report the results")
#  endif
#endif

#ifdef __clang__ 

#  include "clang.hpp"
#  define BOOST_NO_CXX11_THREAD_LOCAL
#  define BOOST_NO_CXX11_ATOMIC_SMART_PTR


#if defined(BOOST_HAS_INT128)
#undef BOOST_HAS_INT128
#endif
#if defined(BOOST_HAS_FLOAT128)
#undef BOOST_HAS_FLOAT128
#endif


#define BOOST_ATOMIC_NO_CMPXCHG16B


#  define BOOST_NO_CWCHAR

#  ifndef __MT__  
#    define BOOST_NO_CXX11_HDR_ATOMIC
#  endif



#define BOOST_NO_FENV_H



#define BOOST_NO_CXX11_HDR_EXCEPTION

#if !defined(_CPPUNWIND) && !defined(__EXCEPTIONS) && !defined(BOOST_NO_EXCEPTIONS)
#  define BOOST_NO_EXCEPTIONS
#endif



#  define BOOST_EMBTC          __CODEGEARC__
#  define BOOST_EMBTC_FULL_VER ((__clang_major__      << 16) | \
(__clang_minor__      <<  8) | \
__clang_patchlevel__         )

#if defined(BOOST_EMBTC)
#  if defined(_WIN64)
#    define BOOST_EMBTC_WIN64 1
#    define BOOST_EMBTC_WINDOWS 1
#    ifndef BOOST_USE_WINDOWS_H
#      define BOOST_USE_WINDOWS_H
#    endif
#  elif defined(_WIN32)
#    define BOOST_EMBTC_WIN32C 1
#    define BOOST_EMBTC_WINDOWS 1
#    ifndef BOOST_USE_WINDOWS_H
#      define BOOST_USE_WINDOWS_H
#    endif
#  elif defined(__APPLE__) && defined(__arm__)
#    define BOOST_EMBTC_IOSARM 1
#    define BOOST_EMBTC_IOS 1
#  elif defined(__APPLE__) && defined(__aarch64__)
#    define BOOST_EMBTC_IOSARM64 1
#    define BOOST_EMBTC_IOS 1
#  elif defined(__ANDROID__) && defined(__arm__)
#    define BOOST_EMBTC_AARM 1
#    define BOOST_EMBTC_ANDROID 1
#  elif
#    if defined(BOOST_ASSERT_CONFIG)
#       error "Unknown Embarcadero driver"
#    else
#       warning "Unknown Embarcadero driver"
#    endif 
#  endif
#endif 

#if defined(BOOST_EMBTC_WINDOWS)

#if !defined(_chdir)
#define _chdir(x) chdir(x)
#endif

#if !defined(_dup2)
#define _dup2(x,y) dup2(x,y)
#endif

#endif

#  undef BOOST_COMPILER
#  define BOOST_COMPILER "Embarcadero-Clang C++ version " BOOST_STRINGIZE(__CODEGEARC__) " clang: " __clang_version__

#else 

# define BOOST_CODEGEARC  __CODEGEARC__
# define BOOST_BORLANDC   __BORLANDC__

#if !defined( BOOST_WITH_CODEGEAR_WARNINGS )
# pragma warn -8004 
# pragma warn -8008 
# pragma warn -8066 
# pragma warn -8104 
# pragma warn -8105 
#endif

#if (__CODEGEARC__ <= 0x613)
#  define BOOST_NO_INTEGRAL_INT64_T
#  define BOOST_NO_DEPENDENT_NESTED_DERIVATIONS
#  define BOOST_NO_PRIVATE_IN_AGGREGATE
#  define BOOST_NO_USING_DECLARATION_OVERLOADS_FROM_TYPENAME_BASE
#  define BOOST_NO_LIMITS_COMPILE_TIME_CONSTANTS
#  define BOOST_SP_NO_SP_CONVERTIBLE
#endif

#if (__CODEGEARC__ <= 0x621)
#  define BOOST_NO_TYPENAME_WITH_CTOR    
#  define BOOST_FUNCTION_SCOPE_USING_DECLARATION_BREAKS_ADL
#  define BOOST_NO_MEMBER_TEMPLATE_FRIENDS
#  define BOOST_NO_NESTED_FRIENDSHIP     
#  define BOOST_NO_USING_TEMPLATE
#  define BOOST_NO_TWO_PHASE_NAME_LOOKUP
#  define BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS

#  define BOOST_NO_COMPLETE_VALUE_INITIALIZATION

#  if defined(NDEBUG) && defined(__cplusplus)
#     include <cstring>
#     undef strcmp
#  endif
#  include <errno.h>
#  ifndef errno
#     define errno errno
#  endif

#endif

#if (__CODEGEARC__ >= 0x620)
#  define BOOST_HAS_PRAGMA_ONCE
#endif

#define BOOST_NO_FENV_H

#if (__CODEGEARC__ <= 0x620)
#define BOOST_NO_CXX11_STATIC_ASSERT
#else
#define BOOST_HAS_STATIC_ASSERT
#endif
#define BOOST_HAS_CHAR16_T
#define BOOST_HAS_CHAR32_T
#define BOOST_HAS_LONG_LONG
#define BOOST_HAS_DECLTYPE
#define BOOST_HAS_EXPLICIT_CONVERSION_OPS
#define BOOST_HAS_SCOPED_ENUM
#define BOOST_HAS_STD_TYPE_TRAITS

#define BOOST_NO_CXX11_AUTO_DECLARATIONS
#define BOOST_NO_CXX11_AUTO_MULTIDECLARATIONS
#define BOOST_NO_CXX11_CONSTEXPR
#define BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
#define BOOST_NO_CXX11_DELETED_FUNCTIONS
#define BOOST_NO_CXX11_EXTERN_TEMPLATE
#define BOOST_NO_CXX11_FUNCTION_TEMPLATE_DEFAULT_ARGS
#define BOOST_NO_CXX11_LAMBDAS
#define BOOST_NO_CXX11_LOCAL_CLASS_TEMPLATE_PARAMETERS
#define BOOST_NO_CXX11_NOEXCEPT
#define BOOST_NO_CXX11_NULLPTR
#define BOOST_NO_CXX11_RANGE_BASED_FOR
#define BOOST_NO_CXX11_RAW_LITERALS
#define BOOST_NO_CXX11_RVALUE_REFERENCES
#define BOOST_NO_SFINAE_EXPR
#define BOOST_NO_CXX11_SFINAE_EXPR
#define BOOST_NO_CXX11_TEMPLATE_ALIASES
#define BOOST_NO_CXX11_UNICODE_LITERALS
#define BOOST_NO_CXX11_VARIADIC_TEMPLATES
#define BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
#define BOOST_NO_CXX11_USER_DEFINED_LITERALS
#define BOOST_NO_CXX11_ALIGNAS
#define BOOST_NO_CXX11_TRAILING_RESULT_TYPES
#define BOOST_NO_CXX11_INLINE_NAMESPACES
#define BOOST_NO_CXX11_REF_QUALIFIERS
#define BOOST_NO_CXX11_FINAL
#define BOOST_NO_CXX11_OVERRIDE
#define BOOST_NO_CXX11_THREAD_LOCAL
#define BOOST_NO_CXX11_DECLTYPE_N3276
#define BOOST_NO_CXX11_UNRESTRICTED_UNION

#if !defined(__cpp_aggregate_nsdmi) || (__cpp_aggregate_nsdmi < 201304)
#  define BOOST_NO_CXX14_AGGREGATE_NSDMI
#endif
#if !defined(__cpp_binary_literals) || (__cpp_binary_literals < 201304)
#  define BOOST_NO_CXX14_BINARY_LITERALS
#endif
#if !defined(__cpp_constexpr) || (__cpp_constexpr < 201304)
#  define BOOST_NO_CXX14_CONSTEXPR
#endif
#if !defined(__cpp_decltype_auto) || (__cpp_decltype_auto < 201304)
#  define BOOST_NO_CXX14_DECLTYPE_AUTO
#endif
#if (__cplusplus < 201304) 
#  define BOOST_NO_CXX14_DIGIT_SEPARATORS
#endif
#if !defined(__cpp_generic_lambdas) || (__cpp_generic_lambdas < 201304)
#  define BOOST_NO_CXX14_GENERIC_LAMBDAS
#endif
#if !defined(__cpp_init_captures) || (__cpp_init_captures < 201304)
#  define BOOST_NO_CXX14_INITIALIZED_LAMBDA_CAPTURES
#endif
#if !defined(__cpp_return_type_deduction) || (__cpp_return_type_deduction < 201304)
#  define BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
#endif
#if !defined(__cpp_variable_templates) || (__cpp_variable_templates < 201304)
#  define BOOST_NO_CXX14_VARIABLE_TEMPLATES
#endif

#if !defined(__cpp_structured_bindings) || (__cpp_structured_bindings < 201606)
#  define BOOST_NO_CXX17_STRUCTURED_BINDINGS
#endif

#if !defined(__cpp_inline_variables) || (__cpp_inline_variables < 201606)
#  define BOOST_NO_CXX17_INLINE_VARIABLES
#endif

#if !defined(__cpp_fold_expressions) || (__cpp_fold_expressions < 201603)
#  define BOOST_NO_CXX17_FOLD_EXPRESSIONS
#endif

#if !defined(__cpp_if_constexpr) || (__cpp_if_constexpr < 201606)
#  define BOOST_NO_CXX17_IF_CONSTEXPR
#endif

#define BOOST_HAS_TR1_HASH
#define BOOST_HAS_TR1_TYPE_TRAITS
#define BOOST_HAS_TR1_UNORDERED_MAP
#define BOOST_HAS_TR1_UNORDERED_SET

#define BOOST_HAS_MACRO_USE_FACET

#define BOOST_NO_CXX11_HDR_INITIALIZER_LIST

#ifdef _WIN32
#  define BOOST_HAS_STDINT_H
#endif

#if !defined(__STRICT_ANSI__)
#  define BOOST_HAS_MS_INT64
#endif
#if !defined(_CPPUNWIND) && !defined(BOOST_CPPUNWIND) && !defined(__EXCEPTIONS) && !defined(BOOST_NO_EXCEPTIONS)
#  define BOOST_NO_EXCEPTIONS
#endif
#if !defined(__STRICT_ANSI__)
#  define BOOST_HAS_DIRENT_H
#endif
#if defined(__STRICT_ANSI__)
#  define BOOST_SYMBOL_EXPORT
#endif
#ifndef BOOST_ABI_PREFIX
#  define BOOST_ABI_PREFIX "boost/config/abi/borland_prefix.hpp"
#endif
#ifndef BOOST_ABI_SUFFIX
#  define BOOST_ABI_SUFFIX "boost/config/abi/borland_suffix.hpp"
#endif
#  pragma defineonoption BOOST_DISABLE_WIN32 -A
#if defined(_MSC_VER) && (_MSC_VER <= 1200)
#  define BOOST_NO_ARGUMENT_DEPENDENT_LOOKUP
#  define BOOST_NO_VOID_RETURNS
#endif

#define BOOST_COMPILER "CodeGear C++ version " BOOST_STRINGIZE(__CODEGEARC__)

#endif 
