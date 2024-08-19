






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


#include "sundials/sundials_nvector.h"


#include "nvector/nvector_mpiplusx.h"

SWIGEXPORT N_Vector _wrap_FN_VMake_MPIPlusX(int const *farg1, N_Vector farg2, void *farg3) {
N_Vector fresult ;
MPI_Comm arg1 ;
N_Vector arg2 = (N_Vector) 0 ;
SUNContext arg3 = (SUNContext) 0 ;
N_Vector result;

#ifdef SUNDIALS_MPI_ENABLED
arg1 = MPI_Comm_f2c((MPI_Fint)(*farg1));
#else
arg1 = *farg1;
#endif
arg2 = (N_Vector)(farg2);
arg3 = (SUNContext)(farg3);
result = (N_Vector)N_VMake_MPIPlusX(arg1,arg2,arg3);
fresult = result;
return fresult;
}


SWIGEXPORT int _wrap_FN_VGetVectorID_MPIPlusX(N_Vector farg1) {
int fresult ;
N_Vector arg1 = (N_Vector) 0 ;
N_Vector_ID result;

arg1 = (N_Vector)(farg1);
result = (N_Vector_ID)N_VGetVectorID_MPIPlusX(arg1);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT double * _wrap_FN_VGetArrayPointer_MPIPlusX(N_Vector farg1) {
double * fresult ;
N_Vector arg1 = (N_Vector) 0 ;
realtype *result = 0 ;

arg1 = (N_Vector)(farg1);
result = (realtype *)N_VGetArrayPointer_MPIPlusX(arg1);
fresult = result;
return fresult;
}


SWIGEXPORT void _wrap_FN_VSetArrayPointer_MPIPlusX(double *farg1, N_Vector farg2) {
realtype *arg1 = (realtype *) 0 ;
N_Vector arg2 = (N_Vector) 0 ;

arg1 = (realtype *)(farg1);
arg2 = (N_Vector)(farg2);
N_VSetArrayPointer_MPIPlusX(arg1,arg2);
}


SWIGEXPORT void _wrap_FN_VPrint_MPIPlusX(N_Vector farg1) {
N_Vector arg1 = (N_Vector) 0 ;

arg1 = (N_Vector)(farg1);
N_VPrint_MPIPlusX(arg1);
}


SWIGEXPORT void _wrap_FN_VPrintFile_MPIPlusX(N_Vector farg1, void *farg2) {
N_Vector arg1 = (N_Vector) 0 ;
FILE *arg2 = (FILE *) 0 ;

arg1 = (N_Vector)(farg1);
arg2 = (FILE *)(farg2);
N_VPrintFile_MPIPlusX(arg1,arg2);
}


SWIGEXPORT N_Vector _wrap_FN_VGetLocalVector_MPIPlusX(N_Vector farg1) {
N_Vector fresult ;
N_Vector arg1 = (N_Vector) 0 ;
N_Vector result;

arg1 = (N_Vector)(farg1);
result = (N_Vector)N_VGetLocalVector_MPIPlusX(arg1);
fresult = result;
return fresult;
}


SWIGEXPORT int64_t _wrap_FN_VGetLocalLength_MPIPlusX(N_Vector farg1) {
int64_t fresult ;
N_Vector arg1 = (N_Vector) 0 ;
sunindextype result;

arg1 = (N_Vector)(farg1);
result = N_VGetLocalLength_MPIPlusX(arg1);
fresult = (sunindextype)(result);
return fresult;
}


SWIGEXPORT int _wrap_FN_VEnableFusedOps_MPIPlusX(N_Vector farg1, int const *farg2) {
int fresult ;
N_Vector arg1 = (N_Vector) 0 ;
int arg2 ;
int result;

arg1 = (N_Vector)(farg1);
arg2 = (int)(*farg2);
result = (int)N_VEnableFusedOps_MPIPlusX(arg1,arg2);
fresult = (int)(result);
return fresult;
}



