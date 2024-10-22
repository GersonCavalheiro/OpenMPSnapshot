






#ifndef SWIGTEMPLATEDISAMBIGUATOR
# if defined(__SUNPRO_CC) && (__SUNPRO_CC <= 0x560)
#  define SWIGTEMPLATEDISAMBIGUATOR template
# elif defined(__HP_aCC)


#  define SWIGTEMPLATEDISAMBIGUATOR template
# else
#  define SWIGTEMPLATEDISAMBIGUATOR
# endif
#endif


#ifndef SWIGINLINE
# if defined(__cplusplus) || (defined(__GNUC__) && !defined(__STRICT_ANSI__))
#   define SWIGINLINE inline
# else
#   define SWIGINLINE
# endif
#endif


#ifndef SWIGUNUSED
# if defined(__GNUC__)
#   if !(defined(__cplusplus)) || (__GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 4))
#     define SWIGUNUSED __attribute__ ((__unused__))
#   else
#     define SWIGUNUSED
#   endif
# elif defined(__ICC)
#   define SWIGUNUSED __attribute__ ((__unused__))
# else
#   define SWIGUNUSED
# endif
#endif

#ifndef SWIG_MSC_UNSUPPRESS_4505
# if defined(_MSC_VER)
#   pragma warning(disable : 4505) 
# endif
#endif

#ifndef SWIGUNUSEDPARM
# ifdef __cplusplus
#   define SWIGUNUSEDPARM(p)
# else
#   define SWIGUNUSEDPARM(p) p SWIGUNUSED
# endif
#endif


#ifndef SWIGINTERN
# define SWIGINTERN static SWIGUNUSED
#endif


#ifndef SWIGINTERNINLINE
# define SWIGINTERNINLINE SWIGINTERN SWIGINLINE
#endif


#ifndef SWIGEXTERN
# ifdef __cplusplus
#   define SWIGEXTERN extern
# else
#   define SWIGEXTERN
# endif
#endif


#if defined(__GNUC__)
#  if (__GNUC__ >= 4) || (__GNUC__ == 3 && __GNUC_MINOR__ >= 4)
#    ifndef GCC_HASCLASSVISIBILITY
#      define GCC_HASCLASSVISIBILITY
#    endif
#  endif
#endif

#ifndef SWIGEXPORT
# if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
#   if defined(STATIC_LINKED)
#     define SWIGEXPORT
#   else
#     define SWIGEXPORT __declspec(dllexport)
#   endif
# else
#   if defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
#     define SWIGEXPORT __attribute__ ((visibility("default")))
#   else
#     define SWIGEXPORT
#   endif
# endif
#endif


#ifndef SWIGSTDCALL
# if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
#   define SWIGSTDCALL __stdcall
# else
#   define SWIGSTDCALL
# endif
#endif


#if !defined(SWIG_NO_CRT_SECURE_NO_DEPRECATE) && defined(_MSC_VER) && !defined(_CRT_SECURE_NO_DEPRECATE)
# define _CRT_SECURE_NO_DEPRECATE
#endif


#if !defined(SWIG_NO_SCL_SECURE_NO_DEPRECATE) && defined(_MSC_VER) && !defined(_SCL_SECURE_NO_DEPRECATE)
# define _SCL_SECURE_NO_DEPRECATE
#endif


#if defined(__APPLE__) && !defined(__ASSERT_MACROS_DEFINE_VERSIONS_WITHOUT_UNDERSCORES)
# define __ASSERT_MACROS_DEFINE_VERSIONS_WITHOUT_UNDERSCORES 0
#endif


#ifdef __INTEL_COMPILER
# pragma warning disable 592
#endif


#define  SWIG_UnknownError    	   -1
#define  SWIG_IOError        	   -2
#define  SWIG_RuntimeError   	   -3
#define  SWIG_IndexError     	   -4
#define  SWIG_TypeError      	   -5
#define  SWIG_DivisionByZero 	   -6
#define  SWIG_OverflowError  	   -7
#define  SWIG_SyntaxError    	   -8
#define  SWIG_ValueError     	   -9
#define  SWIG_SystemError    	   -10
#define  SWIG_AttributeError 	   -11
#define  SWIG_MemoryError    	   -12
#define  SWIG_NullReferenceError   -13




#include <assert.h>
#define SWIG_exception_impl(DECL, CODE, MSG, RETURNNULL) \
{STAN_SUNDIALS_PRINTF("In " DECL ": " MSG); assert(0); RETURNNULL; }


#include <stdio.h>
#if defined(_MSC_VER) || defined(__BORLANDC__) || defined(_WATCOM)
# ifndef snprintf
#  define snprintf _snprintf
# endif
#endif



#define SWIG_contract_assert(RETURNNULL, EXPR, MSG) \
if (!(EXPR)) { SWIG_exception_impl("$decl", SWIG_ValueError, MSG, RETURNNULL); } 


#define SWIGVERSION 0x040000 
#define SWIG_VERSION SWIGVERSION


#define SWIG_as_voidptr(a) (void *)((const void *)(a)) 
#define SWIG_as_voidptrptr(a) ((void)SWIG_as_voidptr(*a),(void**)(a)) 


#include "sundials/sundials_profiler.h"
#if SUNDIALS_MPI_ENABLED
#include <mpi.h>
#endif


#include <stdlib.h>
#ifdef _MSC_VER
# ifndef strtoull
#  define strtoull _strtoui64
# endif
# ifndef strtoll
#  define strtoll _strtoi64
# endif
#endif


typedef struct {
void* data;
size_t size;
} SwigArrayWrapper;


SWIGINTERN SwigArrayWrapper SwigArrayWrapper_uninitialized() {
SwigArrayWrapper result;
result.data = NULL;
result.size = 0;
return result;
}

SWIGEXPORT int _wrap_FSUNProfiler_Free(void *farg1) {
int fresult ;
SUNProfiler *arg1 = (SUNProfiler *) 0 ;
int result;

arg1 = (SUNProfiler *)(farg1);
result = (int)SUNProfiler_Free(arg1);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FSUNProfiler_Begin(void *farg1, SwigArrayWrapper *farg2) {
int fresult ;
SUNProfiler arg1 = (SUNProfiler) 0 ;
char *arg2 = (char *) 0 ;
int result;

arg1 = (SUNProfiler)(farg1);
arg2 = (char *)(farg2->data);
result = (int)SUNProfiler_Begin(arg1,(char const *)arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FSUNProfiler_End(void *farg1, SwigArrayWrapper *farg2) {
int fresult ;
SUNProfiler arg1 = (SUNProfiler) 0 ;
char *arg2 = (char *) 0 ;
int result;

arg1 = (SUNProfiler)(farg1);
arg2 = (char *)(farg2->data);
result = (int)SUNProfiler_End(arg1,(char const *)arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FSUNProfiler_Print(void *farg1, void *farg2) {
int fresult ;
SUNProfiler arg1 = (SUNProfiler) 0 ;
FILE *arg2 = (FILE *) 0 ;
int result;

arg1 = (SUNProfiler)(farg1);
arg2 = (FILE *)(farg2);
result = (int)SUNProfiler_Print(arg1,arg2);
fresult = (int)(result);
return fresult;
}



SWIGEXPORT int _wrap_FSUNProfiler_Create(void *farg1, SwigArrayWrapper *farg2, void *farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
char *arg2 = (char *) 0 ;
SUNProfiler *arg3 = (SUNProfiler *) 0 ;
int result;
#if SUNDIALS_MPI_ENABLED
MPI_Comm comm;
#endif

arg1 = (void *)(farg1);
arg2 = (char *)(farg2->data);
arg3 = (SUNProfiler *)(farg3);
#if SUNDIALS_MPI_ENABLED
if (arg1 != NULL) {
comm = MPI_Comm_f2c(*((MPI_Fint *) arg1));
result = (int)SUNProfiler_Create((void*)&comm,(char const *)arg2,arg3);
}
else {
result = (int)SUNProfiler_Create(arg1,(char const *)arg2,arg3);
}
#else
result = (int)SUNProfiler_Create(arg1,(char const *)arg2,arg3);
#endif
fresult = (int)(result);
return fresult;
}


