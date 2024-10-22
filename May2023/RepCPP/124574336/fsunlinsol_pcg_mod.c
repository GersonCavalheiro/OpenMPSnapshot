






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


#include "sundials/sundials_linearsolver.h"


#include "sunlinsol/sunlinsol_pcg.h"

SWIGEXPORT SUNLinearSolver _wrap_FSUNLinSol_PCG(N_Vector farg1, int const *farg2, int const *farg3, void *farg4) {
SUNLinearSolver fresult ;
N_Vector arg1 = (N_Vector) 0 ;
int arg2 ;
int arg3 ;
SUNContext arg4 = (SUNContext) 0 ;
SUNLinearSolver result;

arg1 = (N_Vector)(farg1);
arg2 = (int)(*farg2);
arg3 = (int)(*farg3);
arg4 = (SUNContext)(farg4);
result = (SUNLinearSolver)SUNLinSol_PCG(arg1,arg2,arg3,arg4);
fresult = result;
return fresult;
}


SWIGEXPORT int _wrap_FSUNLinSol_PCGSetPrecType(SUNLinearSolver farg1, int const *farg2) {
int fresult ;
SUNLinearSolver arg1 = (SUNLinearSolver) 0 ;
int arg2 ;
int result;

arg1 = (SUNLinearSolver)(farg1);
arg2 = (int)(*farg2);
result = (int)SUNLinSol_PCGSetPrecType(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FSUNLinSol_PCGSetMaxl(SUNLinearSolver farg1, int const *farg2) {
int fresult ;
SUNLinearSolver arg1 = (SUNLinearSolver) 0 ;
int arg2 ;
int result;

arg1 = (SUNLinearSolver)(farg1);
arg2 = (int)(*farg2);
result = (int)SUNLinSol_PCGSetMaxl(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FSUNLinSolGetType_PCG(SUNLinearSolver farg1) {
int fresult ;
SUNLinearSolver arg1 = (SUNLinearSolver) 0 ;
SUNLinearSolver_Type result;

arg1 = (SUNLinearSolver)(farg1);
result = (SUNLinearSolver_Type)SUNLinSolGetType_PCG(arg1);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FSUNLinSolGetID_PCG(SUNLinearSolver farg1) {
int fresult ;
SUNLinearSolver arg1 = (SUNLinearSolver) 0 ;
SUNLinearSolver_ID result;

arg1 = (SUNLinearSolver)(farg1);
result = (SUNLinearSolver_ID)SUNLinSolGetID_PCG(arg1);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FSUNLinSolInitialize_PCG(SUNLinearSolver farg1) {
int fresult ;
SUNLinearSolver arg1 = (SUNLinearSolver) 0 ;
int result;

arg1 = (SUNLinearSolver)(farg1);
result = (int)SUNLinSolInitialize_PCG(arg1);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FSUNLinSolSetATimes_PCG(SUNLinearSolver farg1, void *farg2, SUNATimesFn farg3) {
int fresult ;
SUNLinearSolver arg1 = (SUNLinearSolver) 0 ;
void *arg2 = (void *) 0 ;
SUNATimesFn arg3 = (SUNATimesFn) 0 ;
int result;

arg1 = (SUNLinearSolver)(farg1);
arg2 = (void *)(farg2);
arg3 = (SUNATimesFn)(farg3);
result = (int)SUNLinSolSetATimes_PCG(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FSUNLinSolSetPreconditioner_PCG(SUNLinearSolver farg1, void *farg2, SUNPSetupFn farg3, SUNPSolveFn farg4) {
int fresult ;
SUNLinearSolver arg1 = (SUNLinearSolver) 0 ;
void *arg2 = (void *) 0 ;
SUNPSetupFn arg3 = (SUNPSetupFn) 0 ;
SUNPSolveFn arg4 = (SUNPSolveFn) 0 ;
int result;

arg1 = (SUNLinearSolver)(farg1);
arg2 = (void *)(farg2);
arg3 = (SUNPSetupFn)(farg3);
arg4 = (SUNPSolveFn)(farg4);
result = (int)SUNLinSolSetPreconditioner_PCG(arg1,arg2,arg3,arg4);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FSUNLinSolSetScalingVectors_PCG(SUNLinearSolver farg1, N_Vector farg2, N_Vector farg3) {
int fresult ;
SUNLinearSolver arg1 = (SUNLinearSolver) 0 ;
N_Vector arg2 = (N_Vector) 0 ;
N_Vector arg3 = (N_Vector) 0 ;
int result;

arg1 = (SUNLinearSolver)(farg1);
arg2 = (N_Vector)(farg2);
arg3 = (N_Vector)(farg3);
result = (int)SUNLinSolSetScalingVectors_PCG(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FSUNLinSolSetZeroGuess_PCG(SUNLinearSolver farg1, int const *farg2) {
int fresult ;
SUNLinearSolver arg1 = (SUNLinearSolver) 0 ;
int arg2 ;
int result;

arg1 = (SUNLinearSolver)(farg1);
arg2 = (int)(*farg2);
result = (int)SUNLinSolSetZeroGuess_PCG(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FSUNLinSolSetup_PCG(SUNLinearSolver farg1, SUNMatrix farg2) {
int fresult ;
SUNLinearSolver arg1 = (SUNLinearSolver) 0 ;
SUNMatrix arg2 = (SUNMatrix) 0 ;
int result;

arg1 = (SUNLinearSolver)(farg1);
arg2 = (SUNMatrix)(farg2);
result = (int)SUNLinSolSetup_PCG(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FSUNLinSolSolve_PCG(SUNLinearSolver farg1, SUNMatrix farg2, N_Vector farg3, N_Vector farg4, double const *farg5) {
int fresult ;
SUNLinearSolver arg1 = (SUNLinearSolver) 0 ;
SUNMatrix arg2 = (SUNMatrix) 0 ;
N_Vector arg3 = (N_Vector) 0 ;
N_Vector arg4 = (N_Vector) 0 ;
realtype arg5 ;
int result;

arg1 = (SUNLinearSolver)(farg1);
arg2 = (SUNMatrix)(farg2);
arg3 = (N_Vector)(farg3);
arg4 = (N_Vector)(farg4);
arg5 = (realtype)(*farg5);
result = (int)SUNLinSolSolve_PCG(arg1,arg2,arg3,arg4,arg5);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FSUNLinSolNumIters_PCG(SUNLinearSolver farg1) {
int fresult ;
SUNLinearSolver arg1 = (SUNLinearSolver) 0 ;
int result;

arg1 = (SUNLinearSolver)(farg1);
result = (int)SUNLinSolNumIters_PCG(arg1);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT double _wrap_FSUNLinSolResNorm_PCG(SUNLinearSolver farg1) {
double fresult ;
SUNLinearSolver arg1 = (SUNLinearSolver) 0 ;
realtype result;

arg1 = (SUNLinearSolver)(farg1);
result = (realtype)SUNLinSolResNorm_PCG(arg1);
fresult = (realtype)(result);
return fresult;
}


SWIGEXPORT N_Vector _wrap_FSUNLinSolResid_PCG(SUNLinearSolver farg1) {
N_Vector fresult ;
SUNLinearSolver arg1 = (SUNLinearSolver) 0 ;
N_Vector result;

arg1 = (SUNLinearSolver)(farg1);
result = (N_Vector)SUNLinSolResid_PCG(arg1);
fresult = result;
return fresult;
}


SWIGEXPORT int64_t _wrap_FSUNLinSolLastFlag_PCG(SUNLinearSolver farg1) {
int64_t fresult ;
SUNLinearSolver arg1 = (SUNLinearSolver) 0 ;
sunindextype result;

arg1 = (SUNLinearSolver)(farg1);
result = SUNLinSolLastFlag_PCG(arg1);
fresult = (sunindextype)(result);
return fresult;
}


SWIGEXPORT int _wrap_FSUNLinSolSpace_PCG(SUNLinearSolver farg1, long *farg2, long *farg3) {
int fresult ;
SUNLinearSolver arg1 = (SUNLinearSolver) 0 ;
long *arg2 = (long *) 0 ;
long *arg3 = (long *) 0 ;
int result;

arg1 = (SUNLinearSolver)(farg1);
arg2 = (long *)(farg2);
arg3 = (long *)(farg3);
result = (int)SUNLinSolSpace_PCG(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FSUNLinSolFree_PCG(SUNLinearSolver farg1) {
int fresult ;
SUNLinearSolver arg1 = (SUNLinearSolver) 0 ;
int result;

arg1 = (SUNLinearSolver)(farg1);
result = (int)SUNLinSolFree_PCG(arg1);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FSUNLinSolSetInfoFile_PCG(SUNLinearSolver farg1, void *farg2) {
int fresult ;
SUNLinearSolver arg1 = (SUNLinearSolver) 0 ;
FILE *arg2 = (FILE *) 0 ;
int result;

arg1 = (SUNLinearSolver)(farg1);
arg2 = (FILE *)(farg2);
result = (int)SUNLinSolSetInfoFile_PCG(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FSUNLinSolSetPrintLevel_PCG(SUNLinearSolver farg1, int const *farg2) {
int fresult ;
SUNLinearSolver arg1 = (SUNLinearSolver) 0 ;
int arg2 ;
int result;

arg1 = (SUNLinearSolver)(farg1);
arg2 = (int)(*farg2);
result = (int)SUNLinSolSetPrintLevel_PCG(arg1,arg2);
fresult = (int)(result);
return fresult;
}



