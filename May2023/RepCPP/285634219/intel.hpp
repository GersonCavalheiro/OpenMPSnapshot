


#include "boost/config/compiler/common_edg.hpp"

#if defined(__INTEL_COMPILER)
#  define BOOST_INTEL_CXX_VERSION __INTEL_COMPILER
#elif defined(__ICL)
#  define BOOST_INTEL_CXX_VERSION __ICL
#elif defined(__ICC)
#  define BOOST_INTEL_CXX_VERSION __ICC
#elif defined(__ECC)
#  define BOOST_INTEL_CXX_VERSION __ECC
#endif

#define BOOST_COMPILER "Intel C++ version " BOOST_STRINGIZE(BOOST_INTEL_CXX_VERSION)
#define BOOST_INTEL BOOST_INTEL_CXX_VERSION

#if defined(_WIN32) || defined(_WIN64)
#  define BOOST_INTEL_WIN BOOST_INTEL
#else
#  define BOOST_INTEL_LINUX BOOST_INTEL
#endif

#if (BOOST_INTEL_CXX_VERSION <= 500) && defined(_MSC_VER)
#  define BOOST_NO_EXPLICIT_FUNCTION_TEMPLATE_ARGUMENTS
#  define BOOST_NO_TEMPLATE_TEMPLATES
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
#if (defined(__GNUC__) && (__GNUC__ < 4)) || defined(_WIN32) || (BOOST_INTEL_CXX_VERSION <= 1110)
#define BOOST_NO_TWO_PHASE_NAME_LOOKUP
#endif
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

#if _MSC_VER+0 >= 1000
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

#if BOOST_INTEL_CXX_VERSION < 500
#  error "Compiler not supported or configured - please reconfigure"
#endif

#if defined(__APPLE__) && defined(__INTEL_COMPILER)
#  define BOOST_NO_TWO_PHASE_NAME_LOOKUP
#endif

#if defined(__itanium__) && defined(__INTEL_COMPILER)
#  define BOOST_NO_TWO_PHASE_NAME_LOOKUP
#endif

#if (BOOST_INTEL_CXX_VERSION > 1110)
#  if defined(BOOST_ASSERT_CONFIG)
#     error "Unknown compiler version - please run the configure tests and report the results"
#  elif defined(_MSC_VER)
#  endif
#endif

