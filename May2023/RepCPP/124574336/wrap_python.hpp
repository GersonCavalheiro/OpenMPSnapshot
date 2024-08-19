




#ifdef _DEBUG
# ifndef BOOST_DEBUG_PYTHON
#  ifdef _MSC_VER  
#   include <stddef.h>
#   include <stdarg.h>
#   include <stdio.h>
#   include <stdlib.h>
#   include <assert.h>
#   include <errno.h>
#   include <ctype.h>
#   include <wchar.h>
#   include <basetsd.h>
#   include <io.h>
#   include <limits.h>
#   include <float.h>
#   include <string.h>
#   include <math.h>
#   include <time.h>
#  endif
#  undef _DEBUG 
#  define DEBUG_UNDEFINED_FROM_WRAP_PYTHON_H
# endif
#endif

#if defined(__MINGW32__)
# include <cmath>
# include <math.h>
#endif

# include <pyconfig.h>
# if defined(_SGI_COMPILER_VERSION) && _SGI_COMPILER_VERSION >= 740
#  undef _POSIX_C_SOURCE
#  undef _XOPEN_SOURCE
#  undef HAVE_STDINT_H 
# endif

#include <limits.h>
#ifndef ULONG_MAX
# define BOOST_PYTHON_ULONG_MAX_UNDEFINED
#endif
#ifndef LONGLONG_MAX
# define BOOST_PYTHON_LONGLONG_MAX_UNDEFINED
#endif
#ifndef ULONGLONG_MAX
# define BOOST_PYTHON_ULONGLONG_MAX_UNDEFINED
#endif

#include <patchlevel.h>

#if PY_MAJOR_VERSION<2 || PY_MAJOR_VERSION==2 && PY_MINOR_VERSION<2
#error Python 2.2 or higher is required for this version of Boost.Python.
#endif

#if defined(_WIN32) || defined(__CYGWIN__)

# if defined(__GNUC__) && defined(__CYGWIN__)

#  if defined(__LP64__)
#   define SIZEOF_LONG 8
#  else
#   define SIZEOF_LONG 4
#  endif


#  if PY_MAJOR_VERSION < 2 || PY_MAJOR_VERSION == 2 && PY_MINOR_VERSION <= 2

typedef int pid_t;

#   if defined(__LP64__)
#    define WORD_BIT 64
#   else
#    define WORD_BIT 32
#   endif
#   define hypot _hypot
#   include <stdio.h>

#   if PY_MAJOR_VERSION < 2
#    define HAVE_CLOCK
#    define HAVE_STRFTIME
#    define HAVE_STRERROR
#   endif

#   define NT_THREADS

#   ifndef NETSCAPE_PI
#    define USE_SOCKET
#   endif

#   ifdef USE_DL_IMPORT
#    define DL_IMPORT(RTYPE) __declspec(dllimport) RTYPE
#   endif

#   ifdef USE_DL_EXPORT
#    define DL_IMPORT(RTYPE) __declspec(dllexport) RTYPE
#    define DL_EXPORT(RTYPE) __declspec(dllexport) RTYPE
#   endif

#   define HAVE_LONG_LONG 1
#   define LONG_LONG long long
#  endif

# elif defined(__MWERKS__)

#  ifndef _MSC_VER
#   define PY_MSC_VER_DEFINED_FROM_WRAP_PYTHON_H 1
#   define _MSC_VER 900
#  endif

#  undef hypot 

# elif defined(__BORLANDC__) && !defined(__clang__)
#  undef HAVE_HYPOT
#  define HAVE_HYPOT 1
# endif

#endif 

#if defined(__GNUC__)
# if defined(__has_warning)
#  define BOOST_PYTHON_GCC_HAS_WREGISTER __has_warning("-Wregister")
# else
#  define BOOST_PYTHON_GCC_HAS_WREGISTER __GNUC__ >= 7
# endif
#else
# define BOOST_PYTHON_GCC_HAS_WREGISTER 0
#endif

#if BOOST_PYTHON_GCC_HAS_WREGISTER
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wregister"
#elif defined(_MSC_VER)
# pragma warning(push)
# pragma warning(disable : 5033)  
#endif

#if PY_MAJOR_VERSION == 2 && PY_MINOR_VERSION == 2 && PY_MICRO_VERSION < 2
# include <boost/python/detail/python22_fixed.h>
#else
# include <Python.h>
#endif

#if BOOST_PYTHON_GCC_HAS_WREGISTER
# pragma GCC diagnostic pop
#elif defined(_MSC_VER)
# pragma warning(pop)
#endif
#undef BOOST_PYTHON_GCC_HAS_WREGISTER

#ifdef BOOST_PYTHON_ULONG_MAX_UNDEFINED
# undef ULONG_MAX
# undef BOOST_PYTHON_ULONG_MAX_UNDEFINED
#endif

#ifdef BOOST_PYTHON_LONGLONG_MAX_UNDEFINED
# undef LONGLONG_MAX
# undef BOOST_PYTHON_LONGLONG_MAX_UNDEFINED
#endif

#ifdef BOOST_PYTHON_ULONGLONG_MAX_UNDEFINED
# undef ULONGLONG_MAX
# undef BOOST_PYTHON_ULONGLONG_MAX_UNDEFINED
#endif

#ifdef PY_MSC_VER_DEFINED_FROM_WRAP_PYTHON_H
# undef _MSC_VER
#endif

#ifdef DEBUG_UNDEFINED_FROM_WRAP_PYTHON_H
# undef DEBUG_UNDEFINED_FROM_WRAP_PYTHON_H
# define _DEBUG
# ifdef _CRT_NOFORCE_MANIFEST_DEFINED_FROM_WRAP_PYTHON_H
#  undef _CRT_NOFORCE_MANIFEST_DEFINED_FROM_WRAP_PYTHON_H
#  undef _CRT_NOFORCE_MANIFEST
# endif
#endif

#if !defined(PY_MAJOR_VERSION) || PY_MAJOR_VERSION < 2
# define PyObject_INIT(op, typeobj) \
( (op)->ob_type = (typeobj), _Py_NewReference((PyObject *)(op)), (op) )
#endif

#if PY_VERSION_HEX < 0x02060000

# define Py_TYPE(o)    (((PyObject*)(o))->ob_type)
# define Py_REFCNT(o)  (((PyObject*)(o))->ob_refcnt)
# define Py_SIZE(o)    (((PyVarObject*)(o))->ob_size)

# define PyVarObject_HEAD_INIT(type, size) \
PyObject_HEAD_INIT(type) size,

#endif


#ifdef __MWERKS__
# pragma warn_possunwant off
#elif _MSC_VER
# pragma warning(disable:4786)
#endif

#if defined(HAVE_LONG_LONG)
# if defined(PY_LONG_LONG)
#  define BOOST_PYTHON_LONG_LONG PY_LONG_LONG
# elif defined(LONG_LONG)
#  define BOOST_PYTHON_LONG_LONG LONG_LONG
# else
#  error "HAVE_LONG_LONG defined but not PY_LONG_LONG or LONG_LONG"
# endif
#endif
