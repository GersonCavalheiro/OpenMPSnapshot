



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


#include "sundials/sundials_nonlinearsolver.h"

SWIGEXPORT SUNNonlinearSolver _wrap_FSUNNonlinSolNewEmpty(void *farg1) {
SUNNonlinearSolver fresult ;
SUNContext arg1 = (SUNContext) 0 ;
SUNNonlinearSolver result;

arg1 = (SUNContext)(farg1);
result = (SUNNonlinearSolver)SUNNonlinSolNewEmpty(arg1);
fresult = result;
return fresult;
}


SWIGEXPORT void _wrap_FSUNNonlinSolFreeEmpty(SUNNonlinearSolver farg1) {
SUNNonlinearSolver arg1 = (SUNNonlinearSolver) 0 ;

arg1 = (SUNNonlinearSolver)(farg1);
SUNNonlinSolFreeEmpty(arg1);
}


SWIGEXPORT int _wrap_FSUNNonlinSolGetType(SUNNonlinearSolver farg1) {
int fresult ;
SUNNonlinearSolver arg1 = (SUNNonlinearSolver) 0 ;
SUNNonlinearSolver_Type result;

arg1 = (SUNNonlinearSolver)(farg1);
result = (SUNNonlinearSolver_Type)SUNNonlinSolGetType(arg1);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FSUNNonlinSolInitialize(SUNNonlinearSolver farg1) {
int fresult ;
SUNNonlinearSolver arg1 = (SUNNonlinearSolver) 0 ;
int result;

arg1 = (SUNNonlinearSolver)(farg1);
result = (int)SUNNonlinSolInitialize(arg1);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FSUNNonlinSolSetup(SUNNonlinearSolver farg1, N_Vector farg2, void *farg3) {
int fresult ;
SUNNonlinearSolver arg1 = (SUNNonlinearSolver) 0 ;
N_Vector arg2 = (N_Vector) 0 ;
void *arg3 = (void *) 0 ;
int result;

arg1 = (SUNNonlinearSolver)(farg1);
arg2 = (N_Vector)(farg2);
arg3 = (void *)(farg3);
result = (int)SUNNonlinSolSetup(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FSUNNonlinSolSolve(SUNNonlinearSolver farg1, N_Vector farg2, N_Vector farg3, N_Vector farg4, double const *farg5, int const *farg6, void *farg7) {
int fresult ;
SUNNonlinearSolver arg1 = (SUNNonlinearSolver) 0 ;
N_Vector arg2 = (N_Vector) 0 ;
N_Vector arg3 = (N_Vector) 0 ;
N_Vector arg4 = (N_Vector) 0 ;
realtype arg5 ;
int arg6 ;
void *arg7 = (void *) 0 ;
int result;

arg1 = (SUNNonlinearSolver)(farg1);
arg2 = (N_Vector)(farg2);
arg3 = (N_Vector)(farg3);
arg4 = (N_Vector)(farg4);
arg5 = (realtype)(*farg5);
arg6 = (int)(*farg6);
arg7 = (void *)(farg7);
result = (int)SUNNonlinSolSolve(arg1,arg2,arg3,arg4,arg5,arg6,arg7);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FSUNNonlinSolFree(SUNNonlinearSolver farg1) {
int fresult ;
SUNNonlinearSolver arg1 = (SUNNonlinearSolver) 0 ;
int result;

arg1 = (SUNNonlinearSolver)(farg1);
result = (int)SUNNonlinSolFree(arg1);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FSUNNonlinSolSetSysFn(SUNNonlinearSolver farg1, SUNNonlinSolSysFn farg2) {
int fresult ;
SUNNonlinearSolver arg1 = (SUNNonlinearSolver) 0 ;
SUNNonlinSolSysFn arg2 = (SUNNonlinSolSysFn) 0 ;
int result;

arg1 = (SUNNonlinearSolver)(farg1);
arg2 = (SUNNonlinSolSysFn)(farg2);
result = (int)SUNNonlinSolSetSysFn(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FSUNNonlinSolSetLSetupFn(SUNNonlinearSolver farg1, SUNNonlinSolLSetupFn farg2) {
int fresult ;
SUNNonlinearSolver arg1 = (SUNNonlinearSolver) 0 ;
SUNNonlinSolLSetupFn arg2 = (SUNNonlinSolLSetupFn) 0 ;
int result;

arg1 = (SUNNonlinearSolver)(farg1);
arg2 = (SUNNonlinSolLSetupFn)(farg2);
result = (int)SUNNonlinSolSetLSetupFn(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FSUNNonlinSolSetLSolveFn(SUNNonlinearSolver farg1, SUNNonlinSolLSolveFn farg2) {
int fresult ;
SUNNonlinearSolver arg1 = (SUNNonlinearSolver) 0 ;
SUNNonlinSolLSolveFn arg2 = (SUNNonlinSolLSolveFn) 0 ;
int result;

arg1 = (SUNNonlinearSolver)(farg1);
arg2 = (SUNNonlinSolLSolveFn)(farg2);
result = (int)SUNNonlinSolSetLSolveFn(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FSUNNonlinSolSetConvTestFn(SUNNonlinearSolver farg1, SUNNonlinSolConvTestFn farg2, void *farg3) {
int fresult ;
SUNNonlinearSolver arg1 = (SUNNonlinearSolver) 0 ;
SUNNonlinSolConvTestFn arg2 = (SUNNonlinSolConvTestFn) 0 ;
void *arg3 = (void *) 0 ;
int result;

arg1 = (SUNNonlinearSolver)(farg1);
arg2 = (SUNNonlinSolConvTestFn)(farg2);
arg3 = (void *)(farg3);
result = (int)SUNNonlinSolSetConvTestFn(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FSUNNonlinSolSetMaxIters(SUNNonlinearSolver farg1, int const *farg2) {
int fresult ;
SUNNonlinearSolver arg1 = (SUNNonlinearSolver) 0 ;
int arg2 ;
int result;

arg1 = (SUNNonlinearSolver)(farg1);
arg2 = (int)(*farg2);
result = (int)SUNNonlinSolSetMaxIters(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FSUNNonlinSolGetNumIters(SUNNonlinearSolver farg1, long *farg2) {
int fresult ;
SUNNonlinearSolver arg1 = (SUNNonlinearSolver) 0 ;
long *arg2 = (long *) 0 ;
int result;

arg1 = (SUNNonlinearSolver)(farg1);
arg2 = (long *)(farg2);
result = (int)SUNNonlinSolGetNumIters(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FSUNNonlinSolGetCurIter(SUNNonlinearSolver farg1, int *farg2) {
int fresult ;
SUNNonlinearSolver arg1 = (SUNNonlinearSolver) 0 ;
int *arg2 = (int *) 0 ;
int result;

arg1 = (SUNNonlinearSolver)(farg1);
arg2 = (int *)(farg2);
result = (int)SUNNonlinSolGetCurIter(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FSUNNonlinSolGetNumConvFails(SUNNonlinearSolver farg1, long *farg2) {
int fresult ;
SUNNonlinearSolver arg1 = (SUNNonlinearSolver) 0 ;
long *arg2 = (long *) 0 ;
int result;

arg1 = (SUNNonlinearSolver)(farg1);
arg2 = (long *)(farg2);
result = (int)SUNNonlinSolGetNumConvFails(arg1,arg2);
fresult = (int)(result);
return fresult;
}



