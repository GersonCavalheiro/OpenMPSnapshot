


#define BOOST_MSVC _MSC_VER

#if _MSC_FULL_VER > 100000000
#  define BOOST_MSVC_FULL_VER _MSC_FULL_VER
#else
#  define BOOST_MSVC_FULL_VER (_MSC_FULL_VER * 10)
#endif

#pragma warning( disable : 4503 ) 

#if _MSC_VER < 1300  
#  pragma warning( disable : 4786 ) 
#  define BOOST_NO_DEPENDENT_TYPES_IN_TEMPLATE_VALUE_PARAMETERS
#  define BOOST_NO_VOID_RETURNS
#  define BOOST_NO_EXCEPTION_STD_NAMESPACE

#  if BOOST_MSVC == 1202
#    define BOOST_NO_STD_TYPEINFO
#  endif

#endif

#if (_MSC_VER <= 1300)  

#  if !defined(_MSC_EXTENSIONS) && !defined(BOOST_NO_DEPENDENT_TYPES_IN_TEMPLATE_VALUE_PARAMETERS)      
#    define BOOST_NO_DEPENDENT_TYPES_IN_TEMPLATE_VALUE_PARAMETERS
#  endif

#  define BOOST_NO_EXPLICIT_FUNCTION_TEMPLATE_ARGUMENTS
#  define BOOST_NO_INCLASS_MEMBER_INITIALIZATION
#  define BOOST_NO_PRIVATE_IN_AGGREGATE
#  define BOOST_NO_ARGUMENT_DEPENDENT_LOOKUP
#  define BOOST_NO_INTEGRAL_INT64_T
#  define BOOST_NO_DEDUCED_TYPENAME
#  define BOOST_NO_USING_DECLARATION_OVERLOADS_FROM_TYPENAME_BASE

#  define BOOST_NO_MEMBER_TEMPLATES
#  define BOOST_MSVC6_MEMBER_TEMPLATES

#  define BOOST_NO_MEMBER_TEMPLATE_FRIENDS
#  define BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
#  define BOOST_NO_CV_VOID_SPECIALIZATIONS
#  define BOOST_NO_FUNCTION_TEMPLATE_ORDERING
#  define BOOST_NO_USING_TEMPLATE
#  define BOOST_NO_SWPRINTF
#  define BOOST_NO_TEMPLATE_TEMPLATES
#  define BOOST_NO_SFINAE
#  define BOOST_NO_POINTER_TO_MEMBER_TEMPLATE_PARAMETERS
#  define BOOST_NO_IS_ABSTRACT
#  define BOOST_NO_FUNCTION_TYPE_SPECIALIZATIONS
#  if (_MSC_VER > 1200)
#     define BOOST_NO_MEMBER_FUNCTION_SPECIALIZATIONS
#  endif

#endif

#if _MSC_VER < 1400 
#  define BOOST_NO_SWPRINTF
#endif

#if defined(UNDER_CE)
#  define BOOST_NO_SWPRINTF
#endif

#if _MSC_VER <= 1400  
#  define BOOST_NO_MEMBER_TEMPLATE_FRIENDS
#endif

#if _MSC_VER <= 1600  
#  define BOOST_NO_TWO_PHASE_NAME_LOOKUP
#endif

#if _MSC_VER == 1500  
#  define BOOST_NO_ADL_BARRIER
#endif

#if _MSC_VER <= 1500  || !defined(BOOST_STRICT_CONFIG) 
#  define BOOST_NO_INITIALIZER_LISTS
#endif

#ifndef _NATIVE_WCHAR_T_DEFINED
#  define BOOST_NO_INTRINSIC_WCHAR_T
#endif

#if defined(_WIN32_WCE) || defined(UNDER_CE)
#  define BOOST_NO_THREADEX
#  define BOOST_NO_GETSYSTEMTIMEASFILETIME
#  define BOOST_NO_SWPRINTF
#endif

#ifndef _CPPUNWIND 
#  define BOOST_NO_EXCEPTIONS   
#endif 

#if (_MSC_VER >= 1200)
#   define BOOST_HAS_MS_INT64
#endif
#if (_MSC_VER >= 1310) && (defined(_MSC_EXTENSIONS) || (_MSC_VER >= 1500))
#   define BOOST_HAS_LONG_LONG
#else
#   define BOOST_NO_LONG_LONG
#endif
#if (_MSC_VER >= 1400) && !defined(_DEBUG)
#   define BOOST_HAS_NRVO
#endif
#if !defined(_MSC_EXTENSIONS) && !defined(BOOST_DISABLE_WIN32)
#  define BOOST_DISABLE_WIN32
#endif
#if !defined(_CPPRTTI) && !defined(BOOST_NO_RTTI)
#  define BOOST_NO_RTTI
#endif

#define BOOST_HAS_DECLSPEC


#if _MSC_VER < 1600
#define BOOST_NO_AUTO_DECLARATIONS
#define BOOST_NO_AUTO_MULTIDECLARATIONS
#define BOOST_NO_DECLTYPE
#define BOOST_NO_LAMBDAS
#define BOOST_NO_RVALUE_REFERENCES
#define BOOST_NO_STATIC_ASSERT
#endif 

#define BOOST_NO_CHAR16_T
#define BOOST_NO_CHAR32_T
#define BOOST_NO_CONCEPTS
#define BOOST_NO_CONSTEXPR
#define BOOST_NO_DEFAULTED_FUNCTIONS
#define BOOST_NO_DELETED_FUNCTIONS
#define BOOST_NO_EXPLICIT_CONVERSION_OPERATORS
#define BOOST_NO_EXTERN_TEMPLATE
#define BOOST_NO_FUNCTION_TEMPLATE_DEFAULT_ARGS
#define BOOST_NO_INITIALIZER_LISTS
#define BOOST_NO_NULLPTR
#define BOOST_NO_RAW_LITERALS
#define BOOST_NO_SCOPED_ENUMS
#define BOOST_NO_SFINAE_EXPR
#define BOOST_NO_TEMPLATE_ALIASES
#define BOOST_NO_UNICODE_LITERALS
#define BOOST_NO_VARIADIC_TEMPLATES

#ifndef BOOST_ABI_PREFIX
#  define BOOST_ABI_PREFIX "boost/config/abi/msvc_prefix.hpp"
#endif
#ifndef BOOST_ABI_SUFFIX
#  define BOOST_ABI_SUFFIX "boost/config/abi/msvc_suffix.hpp"
#endif

# if defined(UNDER_CE)
#   if _MSC_VER < 1200
#   elif _MSC_VER < 1300 
#     define BOOST_COMPILER_VERSION evc4.0
#   elif _MSC_VER == 1400
#     define BOOST_COMPILER_VERSION evc8
#   elif _MSC_VER == 1500
#     define BOOST_COMPILER_VERSION evc9
#   elif _MSC_VER == 1600
#     define BOOST_COMPILER_VERSION evc10
#   else
#      if defined(BOOST_ASSERT_CONFIG)
#         error "Unknown EVC++ compiler version - please run the configure tests and report the results"
#      else
#         pragma message("Unknown EVC++ compiler version - please run the configure tests and report the results")
#      endif
#   endif
# else
#   if _MSC_VER < 1200
#     define BOOST_COMPILER_VERSION 5.0
#   elif _MSC_VER < 1300
#       define BOOST_COMPILER_VERSION 6.0
#   elif _MSC_VER == 1300
#     define BOOST_COMPILER_VERSION 7.0
#   elif _MSC_VER == 1310
#     define BOOST_COMPILER_VERSION 7.1
#   elif _MSC_VER == 1400
#     define BOOST_COMPILER_VERSION 8.0
#   elif _MSC_VER == 1500
#     define BOOST_COMPILER_VERSION 9.0
#   elif _MSC_VER == 1600
#     define BOOST_COMPILER_VERSION 10.0
#   else
#     define BOOST_COMPILER_VERSION _MSC_VER
#   endif
# endif

#define BOOST_COMPILER "Microsoft Visual C++ version " BOOST_STRINGIZE(BOOST_COMPILER_VERSION)

#if _MSC_VER < 1200
#error "Compiler not supported or configured - please reconfigure"
#endif
#if (_MSC_VER > 1600)
#  if defined(BOOST_ASSERT_CONFIG)
#     error "Unknown compiler version - please run the configure tests and report the results"
#  else
#     pragma message("Unknown compiler version - please run the configure tests and report the results")
#  endif
#endif
