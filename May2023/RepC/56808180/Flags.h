#ifndef __cxxtest__Flags_h__
#define __cxxtest__Flags_h__
#if !defined(CXXTEST_FLAGS)
#   define CXXTEST_FLAGS
#endif 
#if defined(CXXTEST_HAVE_EH) && !defined(_CXXTEST_HAVE_EH)
#   define _CXXTEST_HAVE_EH
#endif 
#if defined(CXXTEST_HAVE_STD) && !defined(_CXXTEST_HAVE_STD)
#   define _CXXTEST_HAVE_STD
#endif 
#if defined(CXXTEST_OLD_TEMPLATE_SYNTAX) && !defined(_CXXTEST_OLD_TEMPLATE_SYNTAX)
#   define _CXXTEST_OLD_TEMPLATE_SYNTAX
#endif 
#if defined(CXXTEST_OLD_STD) && !defined(_CXXTEST_OLD_STD)
#   define _CXXTEST_OLD_STD
#endif 
#if defined(CXXTEST_ABORT_TEST_ON_FAIL) && !defined(_CXXTEST_ABORT_TEST_ON_FAIL)
#   define _CXXTEST_ABORT_TEST_ON_FAIL
#endif 
#if defined(CXXTEST_NO_COPY_CONST) && !defined(_CXXTEST_NO_COPY_CONST)
#   define _CXXTEST_NO_COPY_CONST
#endif 
#if defined(CXXTEST_FACTOR) && !defined(_CXXTEST_FACTOR)
#   define _CXXTEST_FACTOR
#endif 
#if defined(CXXTEST_PARTIAL_TEMPLATE_SPECIALIZATION) && !defined(_CXXTEST_PARTIAL_TEMPLATE_SPECIALIZATION)
#   define _CXXTEST_PARTIAL_TEMPLATE_SPECIALIZATION
#endif 
#if defined(CXXTEST_LONGLONG)
#   if defined(_CXXTEST_LONGLONG)
#       undef _CXXTEST_LONGLONG
#   endif
#   define _CXXTEST_LONGLONG CXXTEST_LONGLONG
#endif 
#ifndef CXXTEST_MAX_DUMP_SIZE
#   define CXXTEST_MAX_DUMP_SIZE 0
#endif 
#if defined(_CXXTEST_ABORT_TEST_ON_FAIL) && !defined(CXXTEST_DEFAULT_ABORT)
#   define CXXTEST_DEFAULT_ABORT true
#endif 
#if !defined(CXXTEST_DEFAULT_ABORT)
#   define CXXTEST_DEFAULT_ABORT false
#endif 
#if defined(_CXXTEST_ABORT_TEST_ON_FAIL) && !defined(_CXXTEST_HAVE_EH)
#   warning "CXXTEST_ABORT_TEST_ON_FAIL is meaningless without CXXTEST_HAVE_EH"
#   undef _CXXTEST_ABORT_TEST_ON_FAIL
#endif 
#ifdef __BORLANDC__
#   if __BORLANDC__ <= 0x520 
#       ifndef _CXXTEST_OLD_STD
#           define _CXXTEST_OLD_STD
#       endif
#       ifndef _CXXTEST_OLD_TEMPLATE_SYNTAX
#           define _CXXTEST_OLD_TEMPLATE_SYNTAX
#       endif
#   endif
#   if __BORLANDC__ >= 0x540 
#       ifndef _CXXTEST_NO_COPY_CONST
#           define _CXXTEST_NO_COPY_CONST
#       endif
#       ifndef _CXXTEST_LONGLONG
#           define _CXXTEST_LONGLONG __int64
#       endif
#   endif
#endif 
#ifdef _MSC_VER 
#   ifndef _CXXTEST_LONGLONG
#       define _CXXTEST_LONGLONG __int64
#   endif
#   if (_MSC_VER >= 0x51E)
#       ifndef _CXXTEST_PARTIAL_TEMPLATE_SPECIALIZATION
#           define _CXXTEST_PARTIAL_TEMPLATE_SPECIALIZATION
#       endif
#   endif
#pragma warning( disable : 4127 )
#pragma warning( disable : 4290 )
#pragma warning( disable : 4511 )
#pragma warning( disable : 4512 )
#pragma warning( disable : 4514 )
#endif 
#ifdef __GNUC__
#   if (__GNUC__ > 2) || (__GNUC__ == 2 && __GNUC_MINOR__ >= 9)
#       ifndef _CXXTEST_PARTIAL_TEMPLATE_SPECIALIZATION
#           define _CXXTEST_PARTIAL_TEMPLATE_SPECIALIZATION
#       endif
#   endif
#   if defined(__LONG_LONG_MAX__) && !defined(__cplusplus)
#      define _CXXTEST_LONGLONG long long
#   endif
#endif 
#ifdef __DMC__ 
#   ifndef _CXXTEST_OLD_STD
#       define _CXXTEST_OLD_STD
#   endif
#endif
#ifdef __SUNPRO_CC 
#   if __SUNPRO_CC >= 0x510
#       ifndef _CXXTEST_PARTIAL_TEMPLATE_SPECIALIZATION
#           define _CXXTEST_PARTIAL_TEMPLATE_SPECIALIZATION
#       endif
#   endif
#endif
#ifdef __xlC__ 
#   if __xlC__ >= 0x0700
#       ifndef _CXXTEST_PARTIAL_TEMPLATE_SPECIALIZATION
#           define _CXXTEST_PARTIAL_TEMPLATE_SPECIALIZATION
#       endif
#   endif
#endif
#endif 
