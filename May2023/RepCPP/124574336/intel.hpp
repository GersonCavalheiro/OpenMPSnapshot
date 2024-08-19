


#if defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 1500) && (defined(_MSC_VER) || defined(__GNUC__))

#ifdef _MSC_VER

#include <boost/config/compiler/visualc.hpp>

#undef BOOST_MSVC
#undef BOOST_MSVC_FULL_VER

#if (__INTEL_COMPILER >= 1500) && (_MSC_VER >= 1900)
#define BOOST_HAS_EXPM1
#define BOOST_HAS_LOG1P
#undef BOOST_NO_CXX14_BINARY_LITERALS
#undef BOOST_NO_SFINAE_EXPR

#endif

#if (__INTEL_COMPILER <= 1600) && !defined(BOOST_NO_CXX14_VARIABLE_TEMPLATES)
#  define BOOST_NO_CXX14_VARIABLE_TEMPLATES
#endif

#else 

#include <boost/config/compiler/gcc.hpp>

#undef BOOST_GCC_VERSION
#undef BOOST_GCC_CXX11
#undef BOOST_GCC
#undef BOOST_FALLTHROUGH

#if (__INTEL_COMPILER <= 1700) && !defined(BOOST_NO_CXX14_CONSTEXPR)
#  define BOOST_NO_CXX14_CONSTEXPR
#endif

#if (__INTEL_COMPILER >= 1800) && (__cplusplus >= 201703)
#  define BOOST_FALLTHROUGH [[fallthrough]]
#endif

#endif 

#undef BOOST_COMPILER

#if defined(__INTEL_COMPILER)
#if __INTEL_COMPILER == 9999
#  define BOOST_INTEL_CXX_VERSION 1200 
#else
#  define BOOST_INTEL_CXX_VERSION __INTEL_COMPILER
#endif
#elif defined(__ICL)
#  define BOOST_INTEL_CXX_VERSION __ICL
#elif defined(__ICC)
#  define BOOST_INTEL_CXX_VERSION __ICC
#elif defined(__ECC)
#  define BOOST_INTEL_CXX_VERSION __ECC
#endif

#if (!(defined(_WIN32) || defined(_WIN64)) && defined(__STDC_HOSTED__) && (__STDC_HOSTED__ && (BOOST_INTEL_CXX_VERSION <= 1200))) || defined(__GXX_EXPERIMENTAL_CPP0X__) || defined(__GXX_EXPERIMENTAL_CXX0X__)
#  define BOOST_INTEL_STDCXX0X
#endif
#if defined(_MSC_VER) && (_MSC_VER >= 1600)
#  define BOOST_INTEL_STDCXX0X
#endif

#ifdef __GNUC__
#  define BOOST_INTEL_GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#endif

#if !defined(BOOST_COMPILER)
#  if defined(BOOST_INTEL_STDCXX0X)
#    define BOOST_COMPILER "Intel C++ C++0x mode version " BOOST_STRINGIZE(BOOST_INTEL_CXX_VERSION)
#  else
#    define BOOST_COMPILER "Intel C++ version " BOOST_STRINGIZE(BOOST_INTEL_CXX_VERSION)
#  endif
#endif

#define BOOST_INTEL BOOST_INTEL_CXX_VERSION

#if defined(_WIN32) || defined(_WIN64)
#  define BOOST_INTEL_WIN BOOST_INTEL
#else
#  define BOOST_INTEL_LINUX BOOST_INTEL
#endif

#else 

#include <boost/config/compiler/common_edg.hpp>

#if defined(__INTEL_COMPILER)
#if __INTEL_COMPILER == 9999
#  define BOOST_INTEL_CXX_VERSION 1200 
#else
#  define BOOST_INTEL_CXX_VERSION __INTEL_COMPILER
#endif
#elif defined(__ICL)
#  define BOOST_INTEL_CXX_VERSION __ICL
#elif defined(__ICC)
#  define BOOST_INTEL_CXX_VERSION __ICC
#elif defined(__ECC)
#  define BOOST_INTEL_CXX_VERSION __ECC
#endif

#if (!(defined(_WIN32) || defined(_WIN64)) && defined(__STDC_HOSTED__) && (__STDC_HOSTED__ && (BOOST_INTEL_CXX_VERSION <= 1200))) || defined(__GXX_EXPERIMENTAL_CPP0X__) || defined(__GXX_EXPERIMENTAL_CXX0X__)
#  define BOOST_INTEL_STDCXX0X
#endif
#if defined(_MSC_VER) && (_MSC_VER >= 1600)
#  define BOOST_INTEL_STDCXX0X
#endif

#ifdef __GNUC__
#  define BOOST_INTEL_GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#endif

#if !defined(BOOST_COMPILER)
#  if defined(BOOST_INTEL_STDCXX0X)
#    define BOOST_COMPILER "Intel C++ C++0x mode version " BOOST_STRINGIZE(BOOST_INTEL_CXX_VERSION)
#  else
#    define BOOST_COMPILER "Intel C++ version " BOOST_STRINGIZE(BOOST_INTEL_CXX_VERSION)
#  endif
#endif

#define BOOST_INTEL BOOST_INTEL_CXX_VERSION

#if defined(_WIN32) || defined(_WIN64)
#  define BOOST_INTEL_WIN BOOST_INTEL
#else
#  define BOOST_INTEL_LINUX BOOST_INTEL
#endif

#if (BOOST_INTEL_CXX_VERSION <= 600)

#  if defined(_MSC_VER) && (_MSC_VER <= 1300) 


#     define BOOST_NO_SWPRINTF
#  endif


#  if defined(_MSC_VER) && (_MSC_VER <= 1200)
#     define BOOST_NO_VOID_RETURNS
#     define BOOST_NO_INTEGRAL_INT64_T
#  endif

#endif

#if (BOOST_INTEL_CXX_VERSION <= 710) && defined(_WIN32)
#  define BOOST_NO_POINTER_TO_MEMBER_TEMPLATE_PARAMETERS
#endif

#if BOOST_INTEL_CXX_VERSION < 600
#  define BOOST_NO_INTRINSIC_WCHAR_T
#else
#  if ((_WCHAR_T_DEFINED + 0) == 0) && ((_WCHAR_T + 0) == 0)
#    define BOOST_NO_INTRINSIC_WCHAR_T
#  endif
#endif

#if defined(__GNUC__) && !defined(BOOST_FUNCTION_SCOPE_USING_DECLARATION_BREAKS_ADL)
#  if ((__GNUC__ == 3) && (__GNUC_MINOR__ <= 2)) || (BOOST_INTEL < 900) || (__INTEL_COMPILER_BUILD_DATE < 20050912)
#     define BOOST_FUNCTION_SCOPE_USING_DECLARATION_BREAKS_ADL
#  endif
#endif
#if (defined(__GNUC__) && (__GNUC__ < 4)) || (defined(_WIN32) && (BOOST_INTEL_CXX_VERSION <= 1200)) || (BOOST_INTEL_CXX_VERSION <= 1200)
#define BOOST_NO_TWO_PHASE_NAME_LOOKUP
#endif
#ifdef __cplusplus
#if defined(BOOST_NO_INTRINSIC_WCHAR_T)
#include <cwchar>
template< typename T > struct assert_no_intrinsic_wchar_t;
template<> struct assert_no_intrinsic_wchar_t<wchar_t> { typedef void type; };
typedef assert_no_intrinsic_wchar_t<unsigned short>::type assert_no_intrinsic_wchar_t_;
#else
template< typename T > struct assert_intrinsic_wchar_t;
template<> struct assert_intrinsic_wchar_t<wchar_t> {};
template<> struct assert_intrinsic_wchar_t<unsigned short> {};
#endif
#endif

#if defined(_MSC_VER) && (_MSC_VER+0 >= 1000)
#  if _MSC_VER >= 1200
#     define BOOST_HAS_MS_INT64
#  endif
#  define BOOST_NO_SWPRINTF
#  define BOOST_NO_TWO_PHASE_NAME_LOOKUP
#elif defined(_WIN32)
#  define BOOST_DISABLE_WIN32
#endif

#if (BOOST_INTEL_CXX_VERSION >= 600)
#  define BOOST_HAS_NRVO
#endif

#if defined(__GNUC__) && BOOST_INTEL_CXX_VERSION >= 800
#define BOOST_LIKELY(x) __builtin_expect(x, 1)
#define BOOST_UNLIKELY(x) __builtin_expect(x, 0)
#endif

#if !defined(__RTTI) && !defined(__INTEL_RTTI__) && !defined(__GXX_RTTI) && !defined(_CPPRTTI)

#if !defined(BOOST_NO_RTTI)
# define BOOST_NO_RTTI
#endif

#if !defined(_MSC_VER) && !defined(BOOST_NO_TYPEID)
# define BOOST_NO_TYPEID
#endif

#endif

#if BOOST_INTEL_CXX_VERSION < 600
#  error "Compiler not supported or configured - please reconfigure"
#endif

#if defined(__APPLE__) && defined(__INTEL_COMPILER)
#  define BOOST_NO_TWO_PHASE_NAME_LOOKUP
#endif

#if defined(__itanium__) && defined(__INTEL_COMPILER)
#  define BOOST_NO_TWO_PHASE_NAME_LOOKUP
#endif

#if defined(__INTEL_COMPILER)
#  if (__INTEL_COMPILER <= 1110) || (__INTEL_COMPILER == 9999) || (defined(_WIN32) && (__INTEL_COMPILER < 1600))
#    define BOOST_NO_COMPLETE_VALUE_INITIALIZATION
#  endif
#endif

#if defined(__GNUC__) && (__GNUC__ >= 4)
#  define BOOST_SYMBOL_EXPORT __attribute__((visibility("default")))
#  define BOOST_SYMBOL_IMPORT
#  define BOOST_SYMBOL_VISIBLE __attribute__((visibility("default")))
#endif

#if defined(__GNUC__) && (BOOST_INTEL_CXX_VERSION >= 1300)
#  define BOOST_MAY_ALIAS __attribute__((__may_alias__))
#endif

#if defined(BOOST_INTEL_STDCXX0X)
#if (BOOST_INTEL_CXX_VERSION >= 1500) && (!defined(BOOST_INTEL_GCC_VERSION) || (BOOST_INTEL_GCC_VERSION >= 40600)) && !defined(_MSC_VER)
#  undef BOOST_NO_CXX11_CONSTEXPR
#endif
#if (BOOST_INTEL_CXX_VERSION >= 1210) && (!defined(BOOST_INTEL_GCC_VERSION) || (BOOST_INTEL_GCC_VERSION >= 40600)) && (!defined(_MSC_VER) || (_MSC_VER >= 1600))
#  undef BOOST_NO_CXX11_NULLPTR
#endif
#if (BOOST_INTEL_CXX_VERSION >= 1210) && (!defined(BOOST_INTEL_GCC_VERSION) || (BOOST_INTEL_GCC_VERSION >= 40700)) && (!defined(_MSC_VER) || (_MSC_FULL_VER >= 180020827))
#  undef BOOST_NO_CXX11_TEMPLATE_ALIASES
#endif

#if (BOOST_INTEL_CXX_VERSION >= 1200) && (!defined(BOOST_INTEL_GCC_VERSION) || (BOOST_INTEL_GCC_VERSION >= 40300)) && (!defined(_MSC_VER) || (_MSC_VER >= 1600))
#  undef BOOST_NO_CXX11_DECLTYPE
#endif

#if (BOOST_INTEL_CXX_VERSION >= 1500) && (!defined(BOOST_INTEL_GCC_VERSION) || (BOOST_INTEL_GCC_VERSION >= 40800)) && (!defined(_MSC_VER) || (_MSC_FULL_VER >= 180020827))
#  undef BOOST_NO_CXX11_DECLTYPE_N3276
#endif

#if (BOOST_INTEL_CXX_VERSION >= 1200) && (!defined(BOOST_INTEL_GCC_VERSION) || (BOOST_INTEL_GCC_VERSION >= 40300)) && (!defined(_MSC_VER) || (_MSC_FULL_VER >= 180020827))
#  undef BOOST_NO_CXX11_FUNCTION_TEMPLATE_DEFAULT_ARGS
#endif

#if (BOOST_INTEL_CXX_VERSION >= 1300) && (!defined(BOOST_INTEL_GCC_VERSION) || (BOOST_INTEL_GCC_VERSION >= 40300)) && (!defined(_MSC_VER) || (_MSC_VER >= 1600))
#  undef BOOST_NO_CXX11_RVALUE_REFERENCES
#endif

#if (BOOST_INTEL_CXX_VERSION >= 1110) && (!defined(BOOST_INTEL_GCC_VERSION) || (BOOST_INTEL_GCC_VERSION >= 40300)) && (!defined(_MSC_VER) || (_MSC_VER >= 1600))
#  undef BOOST_NO_CXX11_STATIC_ASSERT
#endif

#if (BOOST_INTEL_CXX_VERSION >= 1200) && (!defined(BOOST_INTEL_GCC_VERSION) || (BOOST_INTEL_GCC_VERSION >= 40400)) && (!defined(_MSC_VER) || (_MSC_FULL_VER >= 180020827))
#  undef BOOST_NO_CXX11_VARIADIC_TEMPLATES
#endif

#if (BOOST_INTEL_CXX_VERSION >= 1200) && (!defined(BOOST_INTEL_GCC_VERSION) || (BOOST_INTEL_GCC_VERSION >= 40200)) && (!defined(_MSC_VER) || (_MSC_VER >= 1400))
#  undef BOOST_NO_CXX11_VARIADIC_MACROS
#endif

#if (BOOST_INTEL_CXX_VERSION >= 1200) && (!defined(BOOST_INTEL_GCC_VERSION) || (BOOST_INTEL_GCC_VERSION >= 40400)) && (!defined(_MSC_VER) || (_MSC_VER >= 1600))
#  undef BOOST_NO_CXX11_AUTO_DECLARATIONS
#endif

#if (BOOST_INTEL_CXX_VERSION >= 1200) && (!defined(BOOST_INTEL_GCC_VERSION) || (BOOST_INTEL_GCC_VERSION >= 40400)) && (!defined(_MSC_VER) || (_MSC_VER >= 1600))
#  undef BOOST_NO_CXX11_AUTO_MULTIDECLARATIONS
#endif

#if (BOOST_INTEL_CXX_VERSION >= 1400) && (!defined(BOOST_INTEL_GCC_VERSION) || (BOOST_INTEL_GCC_VERSION >= 40400)) && (!defined(_MSC_VER) || (_MSC_VER >= 9999))
#  undef BOOST_NO_CXX11_CHAR16_T
#endif

#if (BOOST_INTEL_CXX_VERSION >= 1400) && (!defined(BOOST_INTEL_GCC_VERSION) || (BOOST_INTEL_GCC_VERSION >= 40400)) && (!defined(_MSC_VER) || (_MSC_VER >= 9999))
#  undef BOOST_NO_CXX11_CHAR32_T
#endif

#if (BOOST_INTEL_CXX_VERSION >= 1200) && (!defined(BOOST_INTEL_GCC_VERSION) || (BOOST_INTEL_GCC_VERSION >= 40400)) && (!defined(_MSC_VER) || (_MSC_FULL_VER >= 180020827))
#  undef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
#endif

#if (BOOST_INTEL_CXX_VERSION >= 1200) && (!defined(BOOST_INTEL_GCC_VERSION) || (BOOST_INTEL_GCC_VERSION >= 40400)) && (!defined(_MSC_VER) || (_MSC_FULL_VER >= 180020827))
#  undef BOOST_NO_CXX11_DELETED_FUNCTIONS
#endif

#if (BOOST_INTEL_CXX_VERSION >= 1400) && (!defined(BOOST_INTEL_GCC_VERSION) || (BOOST_INTEL_GCC_VERSION >= 40400)) && (!defined(_MSC_VER) || (_MSC_VER >= 1700))
#  undef BOOST_NO_CXX11_HDR_INITIALIZER_LIST
#endif

#if (BOOST_INTEL_CXX_VERSION >= 1400) && (!defined(BOOST_INTEL_GCC_VERSION) || (BOOST_INTEL_GCC_VERSION >= 40501)) && (!defined(_MSC_VER) || (_MSC_VER >= 1700))
#  undef BOOST_NO_CXX11_SCOPED_ENUMS
#endif

#if (BOOST_INTEL_CXX_VERSION >= 1200) && (!defined(BOOST_INTEL_GCC_VERSION) || (BOOST_INTEL_GCC_VERSION >= 40500)) && (!defined(_MSC_VER) || (_MSC_VER >= 9999))
#  undef BOOST_NO_SFINAE_EXPR
#endif

#if (BOOST_INTEL_CXX_VERSION >= 1500) && (!defined(BOOST_INTEL_GCC_VERSION) || (BOOST_INTEL_GCC_VERSION >= 40800)) && !defined(_MSC_VER)
#  undef BOOST_NO_CXX11_SFINAE_EXPR
#endif

#if (BOOST_INTEL_CXX_VERSION >= 1500) && (!defined(BOOST_INTEL_GCC_VERSION) || (BOOST_INTEL_GCC_VERSION >= 40500)) && (!defined(_MSC_VER) || (_MSC_FULL_VER >= 180020827))
#  undef BOOST_NO_CXX11_EXPLICIT_CONVERSION_OPERATORS
#endif

#if (BOOST_INTEL_CXX_VERSION >= 1200) && (!defined(BOOST_INTEL_GCC_VERSION) || (BOOST_INTEL_GCC_VERSION >= 40500)) && (!defined(_MSC_VER) || (_MSC_VER >= 1600))
#  undef BOOST_NO_CXX11_LAMBDAS
#endif

#if (BOOST_INTEL_CXX_VERSION >= 1200) && (!defined(BOOST_INTEL_GCC_VERSION) || (BOOST_INTEL_GCC_VERSION >= 40500))
#  undef BOOST_NO_CXX11_LOCAL_CLASS_TEMPLATE_PARAMETERS
#endif

#if (BOOST_INTEL_CXX_VERSION >= 1400) && (!defined(BOOST_INTEL_GCC_VERSION) || (BOOST_INTEL_GCC_VERSION >= 40600)) && (!defined(_MSC_VER) || (_MSC_VER >= 1700))
#  undef BOOST_NO_CXX11_RANGE_BASED_FOR
#endif

#if (BOOST_INTEL_CXX_VERSION >= 1400) && (!defined(BOOST_INTEL_GCC_VERSION) || (BOOST_INTEL_GCC_VERSION >= 40500)) && (!defined(_MSC_VER) || (_MSC_FULL_VER >= 180020827))
#  undef BOOST_NO_CXX11_RAW_LITERALS
#endif

#if (BOOST_INTEL_CXX_VERSION >= 1400) && (!defined(BOOST_INTEL_GCC_VERSION) || (BOOST_INTEL_GCC_VERSION >= 40500)) && (!defined(_MSC_VER) || (_MSC_VER >= 9999))
#  undef BOOST_NO_CXX11_UNICODE_LITERALS
#endif

#if (BOOST_INTEL_CXX_VERSION >= 1500) && (!defined(BOOST_INTEL_GCC_VERSION) || (BOOST_INTEL_GCC_VERSION >= 40600)) && (!defined(_MSC_VER) || (_MSC_VER >= 9999))
#  undef BOOST_NO_CXX11_NOEXCEPT
#endif

#if (BOOST_INTEL_CXX_VERSION >= 1400) && (!defined(BOOST_INTEL_GCC_VERSION) || (BOOST_INTEL_GCC_VERSION >= 40600)) && (!defined(_MSC_VER) || (_MSC_VER >= 9999))
#  undef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
#endif

#if (BOOST_INTEL_CXX_VERSION >= 1500) && (!defined(BOOST_INTEL_GCC_VERSION) || (BOOST_INTEL_GCC_VERSION >= 40700)) && (!defined(_MSC_VER) || (_MSC_FULL_VER >= 190021730))
#  undef BOOST_NO_CXX11_USER_DEFINED_LITERALS
#endif

#if (BOOST_INTEL_CXX_VERSION >= 1500) && (!defined(BOOST_INTEL_GCC_VERSION) || (BOOST_INTEL_GCC_VERSION >= 40800)) && (!defined(_MSC_VER) || (_MSC_FULL_VER >= 190021730))
#  undef BOOST_NO_CXX11_ALIGNAS
#endif

#if (BOOST_INTEL_CXX_VERSION >= 1200) && (!defined(BOOST_INTEL_GCC_VERSION) || (BOOST_INTEL_GCC_VERSION >= 40400)) && (!defined(_MSC_VER) || (_MSC_FULL_VER >= 180020827))
#  undef BOOST_NO_CXX11_TRAILING_RESULT_TYPES
#endif

#if (BOOST_INTEL_CXX_VERSION >= 1400) && (!defined(BOOST_INTEL_GCC_VERSION) || (BOOST_INTEL_GCC_VERSION >= 40400)) && (!defined(_MSC_VER) || (_MSC_FULL_VER >= 190021730))
#  undef BOOST_NO_CXX11_INLINE_NAMESPACES
#endif

#if (BOOST_INTEL_CXX_VERSION >= 1400) && (!defined(BOOST_INTEL_GCC_VERSION) || (BOOST_INTEL_GCC_VERSION >= 40800)) && (!defined(_MSC_VER) || (_MSC_FULL_VER >= 190021730))
#  undef BOOST_NO_CXX11_REF_QUALIFIERS
#endif

#if (BOOST_INTEL_CXX_VERSION >= 1400) && (!defined(BOOST_INTEL_GCC_VERSION) || (BOOST_INTEL_GCC_VERSION >= 40700)) && (!defined(_MSC_VER) || (_MSC_VER >= 1700))
#  undef BOOST_NO_CXX11_FINAL
#  undef BOOST_NO_CXX11_OVERRIDE
#endif

#if (BOOST_INTEL_CXX_VERSION >= 1400) && (!defined(BOOST_INTEL_GCC_VERSION) || (BOOST_INTEL_GCC_VERSION >= 50100)) && (!defined(_MSC_VER))
#  undef BOOST_NO_CXX11_UNRESTRICTED_UNION
#endif

#endif 

#define BOOST_NO_CXX11_FIXED_LENGTH_VARIADIC_TEMPLATE_EXPANSION_PACKS

#if defined(BOOST_INTEL_STDCXX0X) && (BOOST_INTEL_CXX_VERSION <= 1310)
#  define BOOST_NO_CXX11_HDR_FUTURE
#  define BOOST_NO_CXX11_HDR_INITIALIZER_LIST
#endif

#if defined(BOOST_INTEL_STDCXX0X) && (BOOST_INTEL_CXX_VERSION == 1400)
#  define BOOST_NO_CXX11_HDR_FUTURE
#  define BOOST_NO_CXX11_HDR_TUPLE
#endif

#if (BOOST_INTEL_CXX_VERSION < 1200)
#  define BOOST_NO_FENV_H
#endif

#if (BOOST_INTEL_CXX_VERSION <= 1310)
#  define BOOST_NO_CXX11_NON_PUBLIC_DEFAULTED_FUNCTIONS
#endif

#if defined(_MSC_VER) && (_MSC_VER >= 1600)
#  define BOOST_HAS_STDINT_H
#endif

#if defined(__CUDACC__)
#  if defined(BOOST_GCC_CXX11)
#    define BOOST_NVCC_CXX11
#  else
#    define BOOST_NVCC_CXX03
#  endif
#endif

#if defined(__LP64__) && defined(__GNUC__) && (BOOST_INTEL_CXX_VERSION >= 1310) && !defined(BOOST_NVCC_CXX03)
#  define BOOST_HAS_INT128
#endif

#endif 
#if (BOOST_INTEL_CXX_VERSION > 1700)
#  if defined(BOOST_ASSERT_CONFIG)
#     error "Boost.Config is older than your compiler - please check for an updated Boost release."
#  elif defined(_MSC_VER)
#  endif
#endif

