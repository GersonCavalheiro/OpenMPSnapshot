






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


enum {
SWIG_MEM_OWN = 0x01,
SWIG_MEM_RVALUE = 0x02,
SWIG_MEM_CONST = 0x04
};


#define SWIG_check_mutable(SWIG_CLASS_WRAPPER, TYPENAME, FNAME, FUNCNAME, RETURNNULL) \
if ((SWIG_CLASS_WRAPPER).cmemflags & SWIG_MEM_CONST) { \
SWIG_exception_impl(FUNCNAME, SWIG_TypeError, \
"Cannot pass const " TYPENAME " (class " FNAME ") " \
"as a mutable reference", \
RETURNNULL); \
}


#define SWIG_check_nonnull(SWIG_CLASS_WRAPPER, TYPENAME, FNAME, FUNCNAME, RETURNNULL) \
if (!(SWIG_CLASS_WRAPPER).cptr) { \
SWIG_exception_impl(FUNCNAME, SWIG_TypeError, \
"Cannot pass null " TYPENAME " (class " FNAME ") " \
"as a reference", RETURNNULL); \
}


#define SWIG_check_mutable_nonnull(SWIG_CLASS_WRAPPER, TYPENAME, FNAME, FUNCNAME, RETURNNULL) \
SWIG_check_nonnull(SWIG_CLASS_WRAPPER, TYPENAME, FNAME, FUNCNAME, RETURNNULL); \
SWIG_check_mutable(SWIG_CLASS_WRAPPER, TYPENAME, FNAME, FUNCNAME, RETURNNULL);


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


#include "idas/idas.h"
#include "idas/idas_bbdpre.h"
#include "idas/idas_ls.h"


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


#include <string.h>


typedef struct {
void* cptr;
int cmemflags;
} SwigClassWrapper;


SWIGINTERN SwigClassWrapper SwigClassWrapper_uninitialized() {
SwigClassWrapper result;
result.cptr = NULL;
result.cmemflags = 0;
return result;
}


SWIGINTERN void SWIG_assign(SwigClassWrapper* self, SwigClassWrapper other) {
if (self->cptr == NULL) {

if (other.cmemflags & SWIG_MEM_RVALUE) {

self->cptr = other.cptr;
self->cmemflags = other.cmemflags & (~SWIG_MEM_RVALUE);
} else {

self->cptr = other.cptr;
self->cmemflags = other.cmemflags & (~SWIG_MEM_OWN);
}
} else if (other.cptr == NULL) {

free(self->cptr);
*self = SwigClassWrapper_uninitialized();
} else {
if (self->cmemflags & SWIG_MEM_OWN) {
free(self->cptr);
}
self->cptr = other.cptr;
if (other.cmemflags & SWIG_MEM_RVALUE) {

self->cmemflags = other.cmemflags & ~SWIG_MEM_RVALUE;
} else {

self->cmemflags = other.cmemflags & ~SWIG_MEM_OWN;
}
}
}

SWIGEXPORT void * _wrap_FIDACreate(void *farg1) {
void * fresult ;
SUNContext arg1 = (SUNContext) 0 ;
void *result = 0 ;

arg1 = (SUNContext)(farg1);
result = (void *)IDACreate(arg1);
fresult = result;
return fresult;
}


SWIGEXPORT int _wrap_FIDAInit(void *farg1, IDAResFn farg2, double const *farg3, N_Vector farg4, N_Vector farg5) {
int fresult ;
void *arg1 = (void *) 0 ;
IDAResFn arg2 = (IDAResFn) 0 ;
realtype arg3 ;
N_Vector arg4 = (N_Vector) 0 ;
N_Vector arg5 = (N_Vector) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (IDAResFn)(farg2);
arg3 = (realtype)(*farg3);
arg4 = (N_Vector)(farg4);
arg5 = (N_Vector)(farg5);
result = (int)IDAInit(arg1,arg2,arg3,arg4,arg5);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAReInit(void *farg1, double const *farg2, N_Vector farg3, N_Vector farg4) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype arg2 ;
N_Vector arg3 = (N_Vector) 0 ;
N_Vector arg4 = (N_Vector) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype)(*farg2);
arg3 = (N_Vector)(farg3);
arg4 = (N_Vector)(farg4);
result = (int)IDAReInit(arg1,arg2,arg3,arg4);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASStolerances(void *farg1, double const *farg2, double const *farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype arg2 ;
realtype arg3 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype)(*farg2);
arg3 = (realtype)(*farg3);
result = (int)IDASStolerances(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASVtolerances(void *farg1, double const *farg2, N_Vector farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype arg2 ;
N_Vector arg3 = (N_Vector) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype)(*farg2);
arg3 = (N_Vector)(farg3);
result = (int)IDASVtolerances(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAWFtolerances(void *farg1, IDAEwtFn farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
IDAEwtFn arg2 = (IDAEwtFn) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (IDAEwtFn)(farg2);
result = (int)IDAWFtolerances(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDACalcIC(void *farg1, int const *farg2, double const *farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
realtype arg3 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
arg3 = (realtype)(*farg3);
result = (int)IDACalcIC(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetNonlinConvCoefIC(void *farg1, double const *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype arg2 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype)(*farg2);
result = (int)IDASetNonlinConvCoefIC(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetMaxNumStepsIC(void *farg1, int const *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
result = (int)IDASetMaxNumStepsIC(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetMaxNumJacsIC(void *farg1, int const *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
result = (int)IDASetMaxNumJacsIC(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetMaxNumItersIC(void *farg1, int const *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
result = (int)IDASetMaxNumItersIC(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetLineSearchOffIC(void *farg1, int const *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
result = (int)IDASetLineSearchOffIC(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetStepToleranceIC(void *farg1, double const *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype arg2 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype)(*farg2);
result = (int)IDASetStepToleranceIC(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetMaxBacksIC(void *farg1, int const *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
result = (int)IDASetMaxBacksIC(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetErrHandlerFn(void *farg1, IDAErrHandlerFn farg2, void *farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
IDAErrHandlerFn arg2 = (IDAErrHandlerFn) 0 ;
void *arg3 = (void *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (IDAErrHandlerFn)(farg2);
arg3 = (void *)(farg3);
result = (int)IDASetErrHandlerFn(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetErrFile(void *farg1, void *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
FILE *arg2 = (FILE *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (FILE *)(farg2);
result = (int)IDASetErrFile(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetUserData(void *farg1, void *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
void *arg2 = (void *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (void *)(farg2);
result = (int)IDASetUserData(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetMaxOrd(void *farg1, int const *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
result = (int)IDASetMaxOrd(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetMaxNumSteps(void *farg1, long const *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
long arg2 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long)(*farg2);
result = (int)IDASetMaxNumSteps(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetInitStep(void *farg1, double const *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype arg2 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype)(*farg2);
result = (int)IDASetInitStep(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetMaxStep(void *farg1, double const *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype arg2 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype)(*farg2);
result = (int)IDASetMaxStep(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetStopTime(void *farg1, double const *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype arg2 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype)(*farg2);
result = (int)IDASetStopTime(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetNonlinConvCoef(void *farg1, double const *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype arg2 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype)(*farg2);
result = (int)IDASetNonlinConvCoef(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetMaxErrTestFails(void *farg1, int const *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
result = (int)IDASetMaxErrTestFails(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetMaxNonlinIters(void *farg1, int const *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
result = (int)IDASetMaxNonlinIters(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetMaxConvFails(void *farg1, int const *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
result = (int)IDASetMaxConvFails(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetSuppressAlg(void *farg1, int const *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
result = (int)IDASetSuppressAlg(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetId(void *farg1, N_Vector farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
N_Vector arg2 = (N_Vector) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (N_Vector)(farg2);
result = (int)IDASetId(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetConstraints(void *farg1, N_Vector farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
N_Vector arg2 = (N_Vector) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (N_Vector)(farg2);
result = (int)IDASetConstraints(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetNonlinearSolver(void *farg1, SUNNonlinearSolver farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
SUNNonlinearSolver arg2 = (SUNNonlinearSolver) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (SUNNonlinearSolver)(farg2);
result = (int)IDASetNonlinearSolver(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetNlsResFn(void *farg1, IDAResFn farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
IDAResFn arg2 = (IDAResFn) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (IDAResFn)(farg2);
result = (int)IDASetNlsResFn(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDARootInit(void *farg1, int const *farg2, IDARootFn farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
IDARootFn arg3 = (IDARootFn) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
arg3 = (IDARootFn)(farg3);
result = (int)IDARootInit(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetRootDirection(void *farg1, int *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
int *arg2 = (int *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int *)(farg2);
result = (int)IDASetRootDirection(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetNoInactiveRootWarn(void *farg1) {
int fresult ;
void *arg1 = (void *) 0 ;
int result;

arg1 = (void *)(farg1);
result = (int)IDASetNoInactiveRootWarn(arg1);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASolve(void *farg1, double const *farg2, double *farg3, N_Vector farg4, N_Vector farg5, int const *farg6) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype arg2 ;
realtype *arg3 = (realtype *) 0 ;
N_Vector arg4 = (N_Vector) 0 ;
N_Vector arg5 = (N_Vector) 0 ;
int arg6 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype)(*farg2);
arg3 = (realtype *)(farg3);
arg4 = (N_Vector)(farg4);
arg5 = (N_Vector)(farg5);
arg6 = (int)(*farg6);
result = (int)IDASolve(arg1,arg2,arg3,arg4,arg5,arg6);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAComputeY(void *farg1, N_Vector farg2, N_Vector farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
N_Vector arg2 = (N_Vector) 0 ;
N_Vector arg3 = (N_Vector) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (N_Vector)(farg2);
arg3 = (N_Vector)(farg3);
result = (int)IDAComputeY(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAComputeYp(void *farg1, N_Vector farg2, N_Vector farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
N_Vector arg2 = (N_Vector) 0 ;
N_Vector arg3 = (N_Vector) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (N_Vector)(farg2);
arg3 = (N_Vector)(farg3);
result = (int)IDAComputeYp(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAComputeYSens(void *farg1, void *farg2, void *farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
N_Vector *arg2 = (N_Vector *) 0 ;
N_Vector *arg3 = (N_Vector *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (N_Vector *)(farg2);
arg3 = (N_Vector *)(farg3);
result = (int)IDAComputeYSens(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAComputeYpSens(void *farg1, void *farg2, void *farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
N_Vector *arg2 = (N_Vector *) 0 ;
N_Vector *arg3 = (N_Vector *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (N_Vector *)(farg2);
arg3 = (N_Vector *)(farg3);
result = (int)IDAComputeYpSens(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetDky(void *farg1, double const *farg2, int const *farg3, N_Vector farg4) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype arg2 ;
int arg3 ;
N_Vector arg4 = (N_Vector) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype)(*farg2);
arg3 = (int)(*farg3);
arg4 = (N_Vector)(farg4);
result = (int)IDAGetDky(arg1,arg2,arg3,arg4);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetWorkSpace(void *farg1, long *farg2, long *farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
long *arg3 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
arg3 = (long *)(farg3);
result = (int)IDAGetWorkSpace(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetNumSteps(void *farg1, long *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
result = (int)IDAGetNumSteps(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetNumResEvals(void *farg1, long *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
result = (int)IDAGetNumResEvals(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetNumLinSolvSetups(void *farg1, long *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
result = (int)IDAGetNumLinSolvSetups(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetNumErrTestFails(void *farg1, long *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
result = (int)IDAGetNumErrTestFails(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetNumBacktrackOps(void *farg1, long *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
result = (int)IDAGetNumBacktrackOps(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetConsistentIC(void *farg1, N_Vector farg2, N_Vector farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
N_Vector arg2 = (N_Vector) 0 ;
N_Vector arg3 = (N_Vector) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (N_Vector)(farg2);
arg3 = (N_Vector)(farg3);
result = (int)IDAGetConsistentIC(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetLastOrder(void *farg1, int *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
int *arg2 = (int *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int *)(farg2);
result = (int)IDAGetLastOrder(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetCurrentOrder(void *farg1, int *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
int *arg2 = (int *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int *)(farg2);
result = (int)IDAGetCurrentOrder(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetCurrentCj(void *farg1, double *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype *arg2 = (realtype *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype *)(farg2);
result = (int)IDAGetCurrentCj(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetCurrentY(void *farg1, void *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
N_Vector *arg2 = (N_Vector *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (N_Vector *)(farg2);
result = (int)IDAGetCurrentY(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetCurrentYSens(void *farg1, void *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
N_Vector **arg2 = (N_Vector **) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (N_Vector **)(farg2);
result = (int)IDAGetCurrentYSens(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetCurrentYp(void *farg1, void *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
N_Vector *arg2 = (N_Vector *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (N_Vector *)(farg2);
result = (int)IDAGetCurrentYp(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetCurrentYpSens(void *farg1, void *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
N_Vector **arg2 = (N_Vector **) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (N_Vector **)(farg2);
result = (int)IDAGetCurrentYpSens(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetActualInitStep(void *farg1, double *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype *arg2 = (realtype *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype *)(farg2);
result = (int)IDAGetActualInitStep(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetLastStep(void *farg1, double *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype *arg2 = (realtype *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype *)(farg2);
result = (int)IDAGetLastStep(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetCurrentStep(void *farg1, double *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype *arg2 = (realtype *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype *)(farg2);
result = (int)IDAGetCurrentStep(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetCurrentTime(void *farg1, double *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype *arg2 = (realtype *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype *)(farg2);
result = (int)IDAGetCurrentTime(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetTolScaleFactor(void *farg1, double *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype *arg2 = (realtype *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype *)(farg2);
result = (int)IDAGetTolScaleFactor(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetErrWeights(void *farg1, N_Vector farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
N_Vector arg2 = (N_Vector) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (N_Vector)(farg2);
result = (int)IDAGetErrWeights(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetEstLocalErrors(void *farg1, N_Vector farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
N_Vector arg2 = (N_Vector) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (N_Vector)(farg2);
result = (int)IDAGetEstLocalErrors(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetNumGEvals(void *farg1, long *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
result = (int)IDAGetNumGEvals(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetRootInfo(void *farg1, int *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
int *arg2 = (int *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int *)(farg2);
result = (int)IDAGetRootInfo(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetIntegratorStats(void *farg1, long *farg2, long *farg3, long *farg4, long *farg5, int *farg6, int *farg7, double *farg8, double *farg9, double *farg10, double *farg11) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
long *arg3 = (long *) 0 ;
long *arg4 = (long *) 0 ;
long *arg5 = (long *) 0 ;
int *arg6 = (int *) 0 ;
int *arg7 = (int *) 0 ;
realtype *arg8 = (realtype *) 0 ;
realtype *arg9 = (realtype *) 0 ;
realtype *arg10 = (realtype *) 0 ;
realtype *arg11 = (realtype *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
arg3 = (long *)(farg3);
arg4 = (long *)(farg4);
arg5 = (long *)(farg5);
arg6 = (int *)(farg6);
arg7 = (int *)(farg7);
arg8 = (realtype *)(farg8);
arg9 = (realtype *)(farg9);
arg10 = (realtype *)(farg10);
arg11 = (realtype *)(farg11);
result = (int)IDAGetIntegratorStats(arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9,arg10,arg11);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetNonlinearSystemData(void *farg1, double *farg2, void *farg3, void *farg4, void *farg5, void *farg6, void *farg7, double *farg8, void *farg9) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype *arg2 = (realtype *) 0 ;
N_Vector *arg3 = (N_Vector *) 0 ;
N_Vector *arg4 = (N_Vector *) 0 ;
N_Vector *arg5 = (N_Vector *) 0 ;
N_Vector *arg6 = (N_Vector *) 0 ;
N_Vector *arg7 = (N_Vector *) 0 ;
realtype *arg8 = (realtype *) 0 ;
void **arg9 = (void **) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype *)(farg2);
arg3 = (N_Vector *)(farg3);
arg4 = (N_Vector *)(farg4);
arg5 = (N_Vector *)(farg5);
arg6 = (N_Vector *)(farg6);
arg7 = (N_Vector *)(farg7);
arg8 = (realtype *)(farg8);
arg9 = (void **)(farg9);
result = (int)IDAGetNonlinearSystemData(arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetNonlinearSystemDataSens(void *farg1, double *farg2, void *farg3, void *farg4, void *farg5, void *farg6, double *farg7, void *farg8) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype *arg2 = (realtype *) 0 ;
N_Vector **arg3 = (N_Vector **) 0 ;
N_Vector **arg4 = (N_Vector **) 0 ;
N_Vector **arg5 = (N_Vector **) 0 ;
N_Vector **arg6 = (N_Vector **) 0 ;
realtype *arg7 = (realtype *) 0 ;
void **arg8 = (void **) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype *)(farg2);
arg3 = (N_Vector **)(farg3);
arg4 = (N_Vector **)(farg4);
arg5 = (N_Vector **)(farg5);
arg6 = (N_Vector **)(farg6);
arg7 = (realtype *)(farg7);
arg8 = (void **)(farg8);
result = (int)IDAGetNonlinearSystemDataSens(arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetNumNonlinSolvIters(void *farg1, long *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
result = (int)IDAGetNumNonlinSolvIters(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetNumNonlinSolvConvFails(void *farg1, long *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
result = (int)IDAGetNumNonlinSolvConvFails(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetNonlinSolvStats(void *farg1, long *farg2, long *farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
long *arg3 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
arg3 = (long *)(farg3);
result = (int)IDAGetNonlinSolvStats(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT SwigArrayWrapper _wrap_FIDAGetReturnFlagName(long const *farg1) {
SwigArrayWrapper fresult ;
long arg1 ;
char *result = 0 ;

arg1 = (long)(*farg1);
result = (char *)IDAGetReturnFlagName(arg1);
fresult.size = strlen((const char*)(result));
fresult.data = (char *)(result);
return fresult;
}


SWIGEXPORT void _wrap_FIDAFree(void *farg1) {
void **arg1 = (void **) 0 ;

arg1 = (void **)(farg1);
IDAFree(arg1);
}


SWIGEXPORT int _wrap_FIDASetJacTimesResFn(void *farg1, IDAResFn farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
IDAResFn arg2 = (IDAResFn) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (IDAResFn)(farg2);
result = (int)IDASetJacTimesResFn(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAQuadInit(void *farg1, IDAQuadRhsFn farg2, N_Vector farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
IDAQuadRhsFn arg2 = (IDAQuadRhsFn) 0 ;
N_Vector arg3 = (N_Vector) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (IDAQuadRhsFn)(farg2);
arg3 = (N_Vector)(farg3);
result = (int)IDAQuadInit(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAQuadReInit(void *farg1, N_Vector farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
N_Vector arg2 = (N_Vector) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (N_Vector)(farg2);
result = (int)IDAQuadReInit(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAQuadSStolerances(void *farg1, double const *farg2, double const *farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype arg2 ;
realtype arg3 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype)(*farg2);
arg3 = (realtype)(*farg3);
result = (int)IDAQuadSStolerances(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAQuadSVtolerances(void *farg1, double const *farg2, N_Vector farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype arg2 ;
N_Vector arg3 = (N_Vector) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype)(*farg2);
arg3 = (N_Vector)(farg3);
result = (int)IDAQuadSVtolerances(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetQuadErrCon(void *farg1, int const *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
result = (int)IDASetQuadErrCon(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetQuad(void *farg1, double *farg2, N_Vector farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype *arg2 = (realtype *) 0 ;
N_Vector arg3 = (N_Vector) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype *)(farg2);
arg3 = (N_Vector)(farg3);
result = (int)IDAGetQuad(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetQuadDky(void *farg1, double const *farg2, int const *farg3, N_Vector farg4) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype arg2 ;
int arg3 ;
N_Vector arg4 = (N_Vector) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype)(*farg2);
arg3 = (int)(*farg3);
arg4 = (N_Vector)(farg4);
result = (int)IDAGetQuadDky(arg1,arg2,arg3,arg4);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetQuadNumRhsEvals(void *farg1, long *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
result = (int)IDAGetQuadNumRhsEvals(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetQuadNumErrTestFails(void *farg1, long *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
result = (int)IDAGetQuadNumErrTestFails(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetQuadErrWeights(void *farg1, N_Vector farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
N_Vector arg2 = (N_Vector) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (N_Vector)(farg2);
result = (int)IDAGetQuadErrWeights(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetQuadStats(void *farg1, long *farg2, long *farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
long *arg3 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
arg3 = (long *)(farg3);
result = (int)IDAGetQuadStats(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT void _wrap_FIDAQuadFree(void *farg1) {
void *arg1 = (void *) 0 ;

arg1 = (void *)(farg1);
IDAQuadFree(arg1);
}


SWIGEXPORT int _wrap_FIDASensInit(void *farg1, int const *farg2, int const *farg3, IDASensResFn farg4, void *farg5, void *farg6) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
int arg3 ;
IDASensResFn arg4 = (IDASensResFn) 0 ;
N_Vector *arg5 = (N_Vector *) 0 ;
N_Vector *arg6 = (N_Vector *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
arg3 = (int)(*farg3);
arg4 = (IDASensResFn)(farg4);
arg5 = (N_Vector *)(farg5);
arg6 = (N_Vector *)(farg6);
result = (int)IDASensInit(arg1,arg2,arg3,arg4,arg5,arg6);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASensReInit(void *farg1, int const *farg2, void *farg3, void *farg4) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
N_Vector *arg3 = (N_Vector *) 0 ;
N_Vector *arg4 = (N_Vector *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
arg3 = (N_Vector *)(farg3);
arg4 = (N_Vector *)(farg4);
result = (int)IDASensReInit(arg1,arg2,arg3,arg4);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASensSStolerances(void *farg1, double const *farg2, double *farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype arg2 ;
realtype *arg3 = (realtype *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype)(*farg2);
arg3 = (realtype *)(farg3);
result = (int)IDASensSStolerances(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASensSVtolerances(void *farg1, double const *farg2, void *farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype arg2 ;
N_Vector *arg3 = (N_Vector *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype)(*farg2);
arg3 = (N_Vector *)(farg3);
result = (int)IDASensSVtolerances(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASensEEtolerances(void *farg1) {
int fresult ;
void *arg1 = (void *) 0 ;
int result;

arg1 = (void *)(farg1);
result = (int)IDASensEEtolerances(arg1);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetSensConsistentIC(void *farg1, void *farg2, void *farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
N_Vector *arg2 = (N_Vector *) 0 ;
N_Vector *arg3 = (N_Vector *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (N_Vector *)(farg2);
arg3 = (N_Vector *)(farg3);
result = (int)IDAGetSensConsistentIC(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetSensDQMethod(void *farg1, int const *farg2, double const *farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
realtype arg3 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
arg3 = (realtype)(*farg3);
result = (int)IDASetSensDQMethod(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetSensErrCon(void *farg1, int const *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
result = (int)IDASetSensErrCon(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetSensMaxNonlinIters(void *farg1, int const *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
result = (int)IDASetSensMaxNonlinIters(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetSensParams(void *farg1, double *farg2, double *farg3, int *farg4) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype *arg2 = (realtype *) 0 ;
realtype *arg3 = (realtype *) 0 ;
int *arg4 = (int *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype *)(farg2);
arg3 = (realtype *)(farg3);
arg4 = (int *)(farg4);
result = (int)IDASetSensParams(arg1,arg2,arg3,arg4);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetNonlinearSolverSensSim(void *farg1, SUNNonlinearSolver farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
SUNNonlinearSolver arg2 = (SUNNonlinearSolver) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (SUNNonlinearSolver)(farg2);
result = (int)IDASetNonlinearSolverSensSim(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetNonlinearSolverSensStg(void *farg1, SUNNonlinearSolver farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
SUNNonlinearSolver arg2 = (SUNNonlinearSolver) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (SUNNonlinearSolver)(farg2);
result = (int)IDASetNonlinearSolverSensStg(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASensToggleOff(void *farg1) {
int fresult ;
void *arg1 = (void *) 0 ;
int result;

arg1 = (void *)(farg1);
result = (int)IDASensToggleOff(arg1);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetSens(void *farg1, double *farg2, void *farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype *arg2 = (realtype *) 0 ;
N_Vector *arg3 = (N_Vector *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype *)(farg2);
arg3 = (N_Vector *)(farg3);
result = (int)IDAGetSens(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetSens1(void *farg1, double *farg2, int const *farg3, N_Vector farg4) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype *arg2 = (realtype *) 0 ;
int arg3 ;
N_Vector arg4 = (N_Vector) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype *)(farg2);
arg3 = (int)(*farg3);
arg4 = (N_Vector)(farg4);
result = (int)IDAGetSens1(arg1,arg2,arg3,arg4);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetSensDky(void *farg1, double const *farg2, int const *farg3, void *farg4) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype arg2 ;
int arg3 ;
N_Vector *arg4 = (N_Vector *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype)(*farg2);
arg3 = (int)(*farg3);
arg4 = (N_Vector *)(farg4);
result = (int)IDAGetSensDky(arg1,arg2,arg3,arg4);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetSensDky1(void *farg1, double const *farg2, int const *farg3, int const *farg4, N_Vector farg5) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype arg2 ;
int arg3 ;
int arg4 ;
N_Vector arg5 = (N_Vector) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype)(*farg2);
arg3 = (int)(*farg3);
arg4 = (int)(*farg4);
arg5 = (N_Vector)(farg5);
result = (int)IDAGetSensDky1(arg1,arg2,arg3,arg4,arg5);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetSensNumResEvals(void *farg1, long *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
result = (int)IDAGetSensNumResEvals(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetNumResEvalsSens(void *farg1, long *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
result = (int)IDAGetNumResEvalsSens(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetSensNumErrTestFails(void *farg1, long *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
result = (int)IDAGetSensNumErrTestFails(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetSensNumLinSolvSetups(void *farg1, long *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
result = (int)IDAGetSensNumLinSolvSetups(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetSensErrWeights(void *farg1, void *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
N_Vector_S arg2 = (N_Vector_S) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (N_Vector_S)(farg2);
result = (int)IDAGetSensErrWeights(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetSensStats(void *farg1, long *farg2, long *farg3, long *farg4, long *farg5) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
long *arg3 = (long *) 0 ;
long *arg4 = (long *) 0 ;
long *arg5 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
arg3 = (long *)(farg3);
arg4 = (long *)(farg4);
arg5 = (long *)(farg5);
result = (int)IDAGetSensStats(arg1,arg2,arg3,arg4,arg5);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetSensNumNonlinSolvIters(void *farg1, long *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
result = (int)IDAGetSensNumNonlinSolvIters(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetSensNumNonlinSolvConvFails(void *farg1, long *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
result = (int)IDAGetSensNumNonlinSolvConvFails(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetSensNonlinSolvStats(void *farg1, long *farg2, long *farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
long *arg3 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
arg3 = (long *)(farg3);
result = (int)IDAGetSensNonlinSolvStats(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT void _wrap_FIDASensFree(void *farg1) {
void *arg1 = (void *) 0 ;

arg1 = (void *)(farg1);
IDASensFree(arg1);
}


SWIGEXPORT int _wrap_FIDAQuadSensInit(void *farg1, IDAQuadSensRhsFn farg2, void *farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
IDAQuadSensRhsFn arg2 = (IDAQuadSensRhsFn) 0 ;
N_Vector *arg3 = (N_Vector *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (IDAQuadSensRhsFn)(farg2);
arg3 = (N_Vector *)(farg3);
result = (int)IDAQuadSensInit(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAQuadSensReInit(void *farg1, void *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
N_Vector *arg2 = (N_Vector *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (N_Vector *)(farg2);
result = (int)IDAQuadSensReInit(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAQuadSensSStolerances(void *farg1, double const *farg2, double *farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype arg2 ;
realtype *arg3 = (realtype *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype)(*farg2);
arg3 = (realtype *)(farg3);
result = (int)IDAQuadSensSStolerances(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAQuadSensSVtolerances(void *farg1, double const *farg2, void *farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype arg2 ;
N_Vector *arg3 = (N_Vector *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype)(*farg2);
arg3 = (N_Vector *)(farg3);
result = (int)IDAQuadSensSVtolerances(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAQuadSensEEtolerances(void *farg1) {
int fresult ;
void *arg1 = (void *) 0 ;
int result;

arg1 = (void *)(farg1);
result = (int)IDAQuadSensEEtolerances(arg1);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetQuadSensErrCon(void *farg1, int const *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
result = (int)IDASetQuadSensErrCon(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetQuadSens(void *farg1, double *farg2, void *farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype *arg2 = (realtype *) 0 ;
N_Vector *arg3 = (N_Vector *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype *)(farg2);
arg3 = (N_Vector *)(farg3);
result = (int)IDAGetQuadSens(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetQuadSens1(void *farg1, double *farg2, int const *farg3, N_Vector farg4) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype *arg2 = (realtype *) 0 ;
int arg3 ;
N_Vector arg4 = (N_Vector) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype *)(farg2);
arg3 = (int)(*farg3);
arg4 = (N_Vector)(farg4);
result = (int)IDAGetQuadSens1(arg1,arg2,arg3,arg4);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetQuadSensDky(void *farg1, double const *farg2, int const *farg3, void *farg4) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype arg2 ;
int arg3 ;
N_Vector *arg4 = (N_Vector *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype)(*farg2);
arg3 = (int)(*farg3);
arg4 = (N_Vector *)(farg4);
result = (int)IDAGetQuadSensDky(arg1,arg2,arg3,arg4);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetQuadSensDky1(void *farg1, double const *farg2, int const *farg3, int const *farg4, N_Vector farg5) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype arg2 ;
int arg3 ;
int arg4 ;
N_Vector arg5 = (N_Vector) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype)(*farg2);
arg3 = (int)(*farg3);
arg4 = (int)(*farg4);
arg5 = (N_Vector)(farg5);
result = (int)IDAGetQuadSensDky1(arg1,arg2,arg3,arg4,arg5);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetQuadSensNumRhsEvals(void *farg1, long *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
result = (int)IDAGetQuadSensNumRhsEvals(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetQuadSensNumErrTestFails(void *farg1, long *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
result = (int)IDAGetQuadSensNumErrTestFails(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetQuadSensErrWeights(void *farg1, void *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
N_Vector *arg2 = (N_Vector *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (N_Vector *)(farg2);
result = (int)IDAGetQuadSensErrWeights(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetQuadSensStats(void *farg1, long *farg2, long *farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
long *arg3 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
arg3 = (long *)(farg3);
result = (int)IDAGetQuadSensStats(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT void _wrap_FIDAQuadSensFree(void *farg1) {
void *arg1 = (void *) 0 ;

arg1 = (void *)(farg1);
IDAQuadSensFree(arg1);
}


SWIGEXPORT int _wrap_FIDAAdjInit(void *farg1, long const *farg2, int const *farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
long arg2 ;
int arg3 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long)(*farg2);
arg3 = (int)(*farg3);
result = (int)IDAAdjInit(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAAdjReInit(void *farg1) {
int fresult ;
void *arg1 = (void *) 0 ;
int result;

arg1 = (void *)(farg1);
result = (int)IDAAdjReInit(arg1);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT void _wrap_FIDAAdjFree(void *farg1) {
void *arg1 = (void *) 0 ;

arg1 = (void *)(farg1);
IDAAdjFree(arg1);
}


SWIGEXPORT int _wrap_FIDACreateB(void *farg1, int *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
int *arg2 = (int *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int *)(farg2);
result = (int)IDACreateB(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAInitB(void *farg1, int const *farg2, IDAResFnB farg3, double const *farg4, N_Vector farg5, N_Vector farg6) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
IDAResFnB arg3 = (IDAResFnB) 0 ;
realtype arg4 ;
N_Vector arg5 = (N_Vector) 0 ;
N_Vector arg6 = (N_Vector) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
arg3 = (IDAResFnB)(farg3);
arg4 = (realtype)(*farg4);
arg5 = (N_Vector)(farg5);
arg6 = (N_Vector)(farg6);
result = (int)IDAInitB(arg1,arg2,arg3,arg4,arg5,arg6);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAInitBS(void *farg1, int const *farg2, IDAResFnBS farg3, double const *farg4, N_Vector farg5, N_Vector farg6) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
IDAResFnBS arg3 = (IDAResFnBS) 0 ;
realtype arg4 ;
N_Vector arg5 = (N_Vector) 0 ;
N_Vector arg6 = (N_Vector) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
arg3 = (IDAResFnBS)(farg3);
arg4 = (realtype)(*farg4);
arg5 = (N_Vector)(farg5);
arg6 = (N_Vector)(farg6);
result = (int)IDAInitBS(arg1,arg2,arg3,arg4,arg5,arg6);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAReInitB(void *farg1, int const *farg2, double const *farg3, N_Vector farg4, N_Vector farg5) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
realtype arg3 ;
N_Vector arg4 = (N_Vector) 0 ;
N_Vector arg5 = (N_Vector) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
arg3 = (realtype)(*farg3);
arg4 = (N_Vector)(farg4);
arg5 = (N_Vector)(farg5);
result = (int)IDAReInitB(arg1,arg2,arg3,arg4,arg5);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASStolerancesB(void *farg1, int const *farg2, double const *farg3, double const *farg4) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
realtype arg3 ;
realtype arg4 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
arg3 = (realtype)(*farg3);
arg4 = (realtype)(*farg4);
result = (int)IDASStolerancesB(arg1,arg2,arg3,arg4);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASVtolerancesB(void *farg1, int const *farg2, double const *farg3, N_Vector farg4) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
realtype arg3 ;
N_Vector arg4 = (N_Vector) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
arg3 = (realtype)(*farg3);
arg4 = (N_Vector)(farg4);
result = (int)IDASVtolerancesB(arg1,arg2,arg3,arg4);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAQuadInitB(void *farg1, int const *farg2, IDAQuadRhsFnB farg3, N_Vector farg4) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
IDAQuadRhsFnB arg3 = (IDAQuadRhsFnB) 0 ;
N_Vector arg4 = (N_Vector) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
arg3 = (IDAQuadRhsFnB)(farg3);
arg4 = (N_Vector)(farg4);
result = (int)IDAQuadInitB(arg1,arg2,arg3,arg4);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAQuadInitBS(void *farg1, int const *farg2, IDAQuadRhsFnBS farg3, N_Vector farg4) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
IDAQuadRhsFnBS arg3 = (IDAQuadRhsFnBS) 0 ;
N_Vector arg4 = (N_Vector) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
arg3 = (IDAQuadRhsFnBS)(farg3);
arg4 = (N_Vector)(farg4);
result = (int)IDAQuadInitBS(arg1,arg2,arg3,arg4);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAQuadReInitB(void *farg1, int const *farg2, N_Vector farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
N_Vector arg3 = (N_Vector) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
arg3 = (N_Vector)(farg3);
result = (int)IDAQuadReInitB(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAQuadSStolerancesB(void *farg1, int const *farg2, double const *farg3, double const *farg4) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
realtype arg3 ;
realtype arg4 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
arg3 = (realtype)(*farg3);
arg4 = (realtype)(*farg4);
result = (int)IDAQuadSStolerancesB(arg1,arg2,arg3,arg4);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAQuadSVtolerancesB(void *farg1, int const *farg2, double const *farg3, N_Vector farg4) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
realtype arg3 ;
N_Vector arg4 = (N_Vector) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
arg3 = (realtype)(*farg3);
arg4 = (N_Vector)(farg4);
result = (int)IDAQuadSVtolerancesB(arg1,arg2,arg3,arg4);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDACalcICB(void *farg1, int const *farg2, double const *farg3, N_Vector farg4, N_Vector farg5) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
realtype arg3 ;
N_Vector arg4 = (N_Vector) 0 ;
N_Vector arg5 = (N_Vector) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
arg3 = (realtype)(*farg3);
arg4 = (N_Vector)(farg4);
arg5 = (N_Vector)(farg5);
result = (int)IDACalcICB(arg1,arg2,arg3,arg4,arg5);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDACalcICBS(void *farg1, int const *farg2, double const *farg3, N_Vector farg4, N_Vector farg5, void *farg6, void *farg7) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
realtype arg3 ;
N_Vector arg4 = (N_Vector) 0 ;
N_Vector arg5 = (N_Vector) 0 ;
N_Vector *arg6 = (N_Vector *) 0 ;
N_Vector *arg7 = (N_Vector *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
arg3 = (realtype)(*farg3);
arg4 = (N_Vector)(farg4);
arg5 = (N_Vector)(farg5);
arg6 = (N_Vector *)(farg6);
arg7 = (N_Vector *)(farg7);
result = (int)IDACalcICBS(arg1,arg2,arg3,arg4,arg5,arg6,arg7);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASolveF(void *farg1, double const *farg2, double *farg3, N_Vector farg4, N_Vector farg5, int const *farg6, int *farg7) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype arg2 ;
realtype *arg3 = (realtype *) 0 ;
N_Vector arg4 = (N_Vector) 0 ;
N_Vector arg5 = (N_Vector) 0 ;
int arg6 ;
int *arg7 = (int *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype)(*farg2);
arg3 = (realtype *)(farg3);
arg4 = (N_Vector)(farg4);
arg5 = (N_Vector)(farg5);
arg6 = (int)(*farg6);
arg7 = (int *)(farg7);
result = (int)IDASolveF(arg1,arg2,arg3,arg4,arg5,arg6,arg7);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASolveB(void *farg1, double const *farg2, int const *farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype arg2 ;
int arg3 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype)(*farg2);
arg3 = (int)(*farg3);
result = (int)IDASolveB(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAAdjSetNoSensi(void *farg1) {
int fresult ;
void *arg1 = (void *) 0 ;
int result;

arg1 = (void *)(farg1);
result = (int)IDAAdjSetNoSensi(arg1);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetUserDataB(void *farg1, int const *farg2, void *farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
void *arg3 = (void *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
arg3 = (void *)(farg3);
result = (int)IDASetUserDataB(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetMaxOrdB(void *farg1, int const *farg2, int const *farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
int arg3 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
arg3 = (int)(*farg3);
result = (int)IDASetMaxOrdB(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetMaxNumStepsB(void *farg1, int const *farg2, long const *farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
long arg3 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
arg3 = (long)(*farg3);
result = (int)IDASetMaxNumStepsB(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetInitStepB(void *farg1, int const *farg2, double const *farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
realtype arg3 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
arg3 = (realtype)(*farg3);
result = (int)IDASetInitStepB(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetMaxStepB(void *farg1, int const *farg2, double const *farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
realtype arg3 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
arg3 = (realtype)(*farg3);
result = (int)IDASetMaxStepB(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetSuppressAlgB(void *farg1, int const *farg2, int const *farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
int arg3 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
arg3 = (int)(*farg3);
result = (int)IDASetSuppressAlgB(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetIdB(void *farg1, int const *farg2, N_Vector farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
N_Vector arg3 = (N_Vector) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
arg3 = (N_Vector)(farg3);
result = (int)IDASetIdB(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetConstraintsB(void *farg1, int const *farg2, N_Vector farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
N_Vector arg3 = (N_Vector) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
arg3 = (N_Vector)(farg3);
result = (int)IDASetConstraintsB(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetQuadErrConB(void *farg1, int const *farg2, int const *farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
int arg3 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
arg3 = (int)(*farg3);
result = (int)IDASetQuadErrConB(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetNonlinearSolverB(void *farg1, int const *farg2, SUNNonlinearSolver farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
SUNNonlinearSolver arg3 = (SUNNonlinearSolver) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
arg3 = (SUNNonlinearSolver)(farg3);
result = (int)IDASetNonlinearSolverB(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetB(void *farg1, int const *farg2, double *farg3, N_Vector farg4, N_Vector farg5) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
realtype *arg3 = (realtype *) 0 ;
N_Vector arg4 = (N_Vector) 0 ;
N_Vector arg5 = (N_Vector) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
arg3 = (realtype *)(farg3);
arg4 = (N_Vector)(farg4);
arg5 = (N_Vector)(farg5);
result = (int)IDAGetB(arg1,arg2,arg3,arg4,arg5);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetQuadB(void *farg1, int const *farg2, double *farg3, N_Vector farg4) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
realtype *arg3 = (realtype *) 0 ;
N_Vector arg4 = (N_Vector) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
arg3 = (realtype *)(farg3);
arg4 = (N_Vector)(farg4);
result = (int)IDAGetQuadB(arg1,arg2,arg3,arg4);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT void * _wrap_FIDAGetAdjIDABmem(void *farg1, int const *farg2) {
void * fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
void *result = 0 ;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
result = (void *)IDAGetAdjIDABmem(arg1,arg2);
fresult = result;
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetConsistentICB(void *farg1, int const *farg2, N_Vector farg3, N_Vector farg4) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
N_Vector arg3 = (N_Vector) 0 ;
N_Vector arg4 = (N_Vector) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
arg3 = (N_Vector)(farg3);
arg4 = (N_Vector)(farg4);
result = (int)IDAGetConsistentICB(arg1,arg2,arg3,arg4);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetAdjY(void *farg1, double const *farg2, N_Vector farg3, N_Vector farg4) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype arg2 ;
N_Vector arg3 = (N_Vector) 0 ;
N_Vector arg4 = (N_Vector) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype)(*farg2);
arg3 = (N_Vector)(farg3);
arg4 = (N_Vector)(farg4);
result = (int)IDAGetAdjY(arg1,arg2,arg3,arg4);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT void _wrap_IDAadjCheckPointRec_my_addr_set(SwigClassWrapper const *farg1, void *farg2) {
IDAadjCheckPointRec *arg1 = (IDAadjCheckPointRec *) 0 ;
void *arg2 = (void *) 0 ;

SWIG_check_mutable_nonnull(*farg1, "IDAadjCheckPointRec *", "IDAadjCheckPointRec", "IDAadjCheckPointRec::my_addr", return );
arg1 = (IDAadjCheckPointRec *)(farg1->cptr);
arg2 = (void *)(farg2);
if (arg1) (arg1)->my_addr = arg2;
}


SWIGEXPORT void * _wrap_IDAadjCheckPointRec_my_addr_get(SwigClassWrapper const *farg1) {
void * fresult ;
IDAadjCheckPointRec *arg1 = (IDAadjCheckPointRec *) 0 ;
void *result = 0 ;

SWIG_check_mutable_nonnull(*farg1, "IDAadjCheckPointRec *", "IDAadjCheckPointRec", "IDAadjCheckPointRec::my_addr", return 0);
arg1 = (IDAadjCheckPointRec *)(farg1->cptr);
result = (void *) ((arg1)->my_addr);
fresult = result;
return fresult;
}


SWIGEXPORT void _wrap_IDAadjCheckPointRec_next_addr_set(SwigClassWrapper const *farg1, void *farg2) {
IDAadjCheckPointRec *arg1 = (IDAadjCheckPointRec *) 0 ;
void *arg2 = (void *) 0 ;

SWIG_check_mutable_nonnull(*farg1, "IDAadjCheckPointRec *", "IDAadjCheckPointRec", "IDAadjCheckPointRec::next_addr", return );
arg1 = (IDAadjCheckPointRec *)(farg1->cptr);
arg2 = (void *)(farg2);
if (arg1) (arg1)->next_addr = arg2;
}


SWIGEXPORT void * _wrap_IDAadjCheckPointRec_next_addr_get(SwigClassWrapper const *farg1) {
void * fresult ;
IDAadjCheckPointRec *arg1 = (IDAadjCheckPointRec *) 0 ;
void *result = 0 ;

SWIG_check_mutable_nonnull(*farg1, "IDAadjCheckPointRec *", "IDAadjCheckPointRec", "IDAadjCheckPointRec::next_addr", return 0);
arg1 = (IDAadjCheckPointRec *)(farg1->cptr);
result = (void *) ((arg1)->next_addr);
fresult = result;
return fresult;
}


SWIGEXPORT void _wrap_IDAadjCheckPointRec_t0_set(SwigClassWrapper const *farg1, double const *farg2) {
IDAadjCheckPointRec *arg1 = (IDAadjCheckPointRec *) 0 ;
realtype arg2 ;

SWIG_check_mutable_nonnull(*farg1, "IDAadjCheckPointRec *", "IDAadjCheckPointRec", "IDAadjCheckPointRec::t0", return );
arg1 = (IDAadjCheckPointRec *)(farg1->cptr);
arg2 = (realtype)(*farg2);
if (arg1) (arg1)->t0 = arg2;
}


SWIGEXPORT double _wrap_IDAadjCheckPointRec_t0_get(SwigClassWrapper const *farg1) {
double fresult ;
IDAadjCheckPointRec *arg1 = (IDAadjCheckPointRec *) 0 ;
realtype result;

SWIG_check_mutable_nonnull(*farg1, "IDAadjCheckPointRec *", "IDAadjCheckPointRec", "IDAadjCheckPointRec::t0", return 0);
arg1 = (IDAadjCheckPointRec *)(farg1->cptr);
result = (realtype) ((arg1)->t0);
fresult = (realtype)(result);
return fresult;
}


SWIGEXPORT void _wrap_IDAadjCheckPointRec_t1_set(SwigClassWrapper const *farg1, double const *farg2) {
IDAadjCheckPointRec *arg1 = (IDAadjCheckPointRec *) 0 ;
realtype arg2 ;

SWIG_check_mutable_nonnull(*farg1, "IDAadjCheckPointRec *", "IDAadjCheckPointRec", "IDAadjCheckPointRec::t1", return );
arg1 = (IDAadjCheckPointRec *)(farg1->cptr);
arg2 = (realtype)(*farg2);
if (arg1) (arg1)->t1 = arg2;
}


SWIGEXPORT double _wrap_IDAadjCheckPointRec_t1_get(SwigClassWrapper const *farg1) {
double fresult ;
IDAadjCheckPointRec *arg1 = (IDAadjCheckPointRec *) 0 ;
realtype result;

SWIG_check_mutable_nonnull(*farg1, "IDAadjCheckPointRec *", "IDAadjCheckPointRec", "IDAadjCheckPointRec::t1", return 0);
arg1 = (IDAadjCheckPointRec *)(farg1->cptr);
result = (realtype) ((arg1)->t1);
fresult = (realtype)(result);
return fresult;
}


SWIGEXPORT void _wrap_IDAadjCheckPointRec_nstep_set(SwigClassWrapper const *farg1, long const *farg2) {
IDAadjCheckPointRec *arg1 = (IDAadjCheckPointRec *) 0 ;
long arg2 ;

SWIG_check_mutable_nonnull(*farg1, "IDAadjCheckPointRec *", "IDAadjCheckPointRec", "IDAadjCheckPointRec::nstep", return );
arg1 = (IDAadjCheckPointRec *)(farg1->cptr);
arg2 = (long)(*farg2);
if (arg1) (arg1)->nstep = arg2;
}


SWIGEXPORT long _wrap_IDAadjCheckPointRec_nstep_get(SwigClassWrapper const *farg1) {
long fresult ;
IDAadjCheckPointRec *arg1 = (IDAadjCheckPointRec *) 0 ;
long result;

SWIG_check_mutable_nonnull(*farg1, "IDAadjCheckPointRec *", "IDAadjCheckPointRec", "IDAadjCheckPointRec::nstep", return 0);
arg1 = (IDAadjCheckPointRec *)(farg1->cptr);
result = (long) ((arg1)->nstep);
fresult = (long)(result);
return fresult;
}


SWIGEXPORT void _wrap_IDAadjCheckPointRec_order_set(SwigClassWrapper const *farg1, int const *farg2) {
IDAadjCheckPointRec *arg1 = (IDAadjCheckPointRec *) 0 ;
int arg2 ;

SWIG_check_mutable_nonnull(*farg1, "IDAadjCheckPointRec *", "IDAadjCheckPointRec", "IDAadjCheckPointRec::order", return );
arg1 = (IDAadjCheckPointRec *)(farg1->cptr);
arg2 = (int)(*farg2);
if (arg1) (arg1)->order = arg2;
}


SWIGEXPORT int _wrap_IDAadjCheckPointRec_order_get(SwigClassWrapper const *farg1) {
int fresult ;
IDAadjCheckPointRec *arg1 = (IDAadjCheckPointRec *) 0 ;
int result;

SWIG_check_mutable_nonnull(*farg1, "IDAadjCheckPointRec *", "IDAadjCheckPointRec", "IDAadjCheckPointRec::order", return 0);
arg1 = (IDAadjCheckPointRec *)(farg1->cptr);
result = (int) ((arg1)->order);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT void _wrap_IDAadjCheckPointRec_step_set(SwigClassWrapper const *farg1, double const *farg2) {
IDAadjCheckPointRec *arg1 = (IDAadjCheckPointRec *) 0 ;
realtype arg2 ;

SWIG_check_mutable_nonnull(*farg1, "IDAadjCheckPointRec *", "IDAadjCheckPointRec", "IDAadjCheckPointRec::step", return );
arg1 = (IDAadjCheckPointRec *)(farg1->cptr);
arg2 = (realtype)(*farg2);
if (arg1) (arg1)->step = arg2;
}


SWIGEXPORT double _wrap_IDAadjCheckPointRec_step_get(SwigClassWrapper const *farg1) {
double fresult ;
IDAadjCheckPointRec *arg1 = (IDAadjCheckPointRec *) 0 ;
realtype result;

SWIG_check_mutable_nonnull(*farg1, "IDAadjCheckPointRec *", "IDAadjCheckPointRec", "IDAadjCheckPointRec::step", return 0);
arg1 = (IDAadjCheckPointRec *)(farg1->cptr);
result = (realtype) ((arg1)->step);
fresult = (realtype)(result);
return fresult;
}


SWIGEXPORT SwigClassWrapper _wrap_new_IDAadjCheckPointRec() {
SwigClassWrapper fresult ;
IDAadjCheckPointRec *result = 0 ;

result = (IDAadjCheckPointRec *)calloc(1, sizeof(IDAadjCheckPointRec));
fresult.cptr = result;
fresult.cmemflags = SWIG_MEM_RVALUE | (1 ? SWIG_MEM_OWN : 0);
return fresult;
}


SWIGEXPORT void _wrap_delete_IDAadjCheckPointRec(SwigClassWrapper *farg1) {
IDAadjCheckPointRec *arg1 = (IDAadjCheckPointRec *) 0 ;

SWIG_check_mutable(*farg1, "IDAadjCheckPointRec *", "IDAadjCheckPointRec", "IDAadjCheckPointRec::~IDAadjCheckPointRec()", return );
arg1 = (IDAadjCheckPointRec *)(farg1->cptr);
free((char *) arg1);
}


SWIGEXPORT void _wrap_IDAadjCheckPointRec_op_assign__(SwigClassWrapper *farg1, SwigClassWrapper const *farg2) {
IDAadjCheckPointRec *arg1 = (IDAadjCheckPointRec *) 0 ;
IDAadjCheckPointRec *arg2 = 0 ;

(void)sizeof(arg1);
(void)sizeof(arg2);
SWIG_assign(farg1, *farg2);

}


SWIGEXPORT int _wrap_FIDAGetAdjCheckPointsInfo(void *farg1, SwigClassWrapper const *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
IDAadjCheckPointRec *arg2 = (IDAadjCheckPointRec *) 0 ;
int result;

arg1 = (void *)(farg1);
SWIG_check_mutable(*farg2, "IDAadjCheckPointRec *", "IDAadjCheckPointRec", "IDAGetAdjCheckPointsInfo(void *,IDAadjCheckPointRec *)", return 0);
arg2 = (IDAadjCheckPointRec *)(farg2->cptr);
result = (int)IDAGetAdjCheckPointsInfo(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetJacTimesResFnB(void *farg1, int const *farg2, IDAResFn farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
IDAResFn arg3 = (IDAResFn) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
arg3 = (IDAResFn)(farg3);
result = (int)IDASetJacTimesResFnB(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetAdjDataPointHermite(void *farg1, int const *farg2, double *farg3, N_Vector farg4, N_Vector farg5) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
realtype *arg3 = (realtype *) 0 ;
N_Vector arg4 = (N_Vector) 0 ;
N_Vector arg5 = (N_Vector) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
arg3 = (realtype *)(farg3);
arg4 = (N_Vector)(farg4);
arg5 = (N_Vector)(farg5);
result = (int)IDAGetAdjDataPointHermite(arg1,arg2,arg3,arg4,arg5);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetAdjDataPointPolynomial(void *farg1, int const *farg2, double *farg3, int *farg4, N_Vector farg5) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
realtype *arg3 = (realtype *) 0 ;
int *arg4 = (int *) 0 ;
N_Vector arg5 = (N_Vector) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
arg3 = (realtype *)(farg3);
arg4 = (int *)(farg4);
arg5 = (N_Vector)(farg5);
result = (int)IDAGetAdjDataPointPolynomial(arg1,arg2,arg3,arg4,arg5);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetAdjCurrentCheckPoint(void *farg1, void *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
void **arg2 = (void **) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (void **)(farg2);
result = (int)IDAGetAdjCurrentCheckPoint(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDABBDPrecInit(void *farg1, int64_t const *farg2, int64_t const *farg3, int64_t const *farg4, int64_t const *farg5, int64_t const *farg6, double const *farg7, IDABBDLocalFn farg8, IDABBDCommFn farg9) {
int fresult ;
void *arg1 = (void *) 0 ;
sunindextype arg2 ;
sunindextype arg3 ;
sunindextype arg4 ;
sunindextype arg5 ;
sunindextype arg6 ;
realtype arg7 ;
IDABBDLocalFn arg8 = (IDABBDLocalFn) 0 ;
IDABBDCommFn arg9 = (IDABBDCommFn) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (sunindextype)(*farg2);
arg3 = (sunindextype)(*farg3);
arg4 = (sunindextype)(*farg4);
arg5 = (sunindextype)(*farg5);
arg6 = (sunindextype)(*farg6);
arg7 = (realtype)(*farg7);
arg8 = (IDABBDLocalFn)(farg8);
arg9 = (IDABBDCommFn)(farg9);
result = (int)IDABBDPrecInit(arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDABBDPrecReInit(void *farg1, int64_t const *farg2, int64_t const *farg3, double const *farg4) {
int fresult ;
void *arg1 = (void *) 0 ;
sunindextype arg2 ;
sunindextype arg3 ;
realtype arg4 ;
int result;

arg1 = (void *)(farg1);
arg2 = (sunindextype)(*farg2);
arg3 = (sunindextype)(*farg3);
arg4 = (realtype)(*farg4);
result = (int)IDABBDPrecReInit(arg1,arg2,arg3,arg4);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDABBDPrecGetWorkSpace(void *farg1, long *farg2, long *farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
long *arg3 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
arg3 = (long *)(farg3);
result = (int)IDABBDPrecGetWorkSpace(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDABBDPrecGetNumGfnEvals(void *farg1, long *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
result = (int)IDABBDPrecGetNumGfnEvals(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDABBDPrecInitB(void *farg1, int const *farg2, int64_t const *farg3, int64_t const *farg4, int64_t const *farg5, int64_t const *farg6, int64_t const *farg7, double const *farg8, IDABBDLocalFnB farg9, IDABBDCommFnB farg10) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
sunindextype arg3 ;
sunindextype arg4 ;
sunindextype arg5 ;
sunindextype arg6 ;
sunindextype arg7 ;
realtype arg8 ;
IDABBDLocalFnB arg9 = (IDABBDLocalFnB) 0 ;
IDABBDCommFnB arg10 = (IDABBDCommFnB) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
arg3 = (sunindextype)(*farg3);
arg4 = (sunindextype)(*farg4);
arg5 = (sunindextype)(*farg5);
arg6 = (sunindextype)(*farg6);
arg7 = (sunindextype)(*farg7);
arg8 = (realtype)(*farg8);
arg9 = (IDABBDLocalFnB)(farg9);
arg10 = (IDABBDCommFnB)(farg10);
result = (int)IDABBDPrecInitB(arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9,arg10);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDABBDPrecReInitB(void *farg1, int const *farg2, int64_t const *farg3, int64_t const *farg4, double const *farg5) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
sunindextype arg3 ;
sunindextype arg4 ;
realtype arg5 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
arg3 = (sunindextype)(*farg3);
arg4 = (sunindextype)(*farg4);
arg5 = (realtype)(*farg5);
result = (int)IDABBDPrecReInitB(arg1,arg2,arg3,arg4,arg5);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetLinearSolver(void *farg1, SUNLinearSolver farg2, SUNMatrix farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
SUNLinearSolver arg2 = (SUNLinearSolver) 0 ;
SUNMatrix arg3 = (SUNMatrix) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (SUNLinearSolver)(farg2);
arg3 = (SUNMatrix)(farg3);
result = (int)IDASetLinearSolver(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetJacFn(void *farg1, IDALsJacFn farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
IDALsJacFn arg2 = (IDALsJacFn) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (IDALsJacFn)(farg2);
result = (int)IDASetJacFn(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetPreconditioner(void *farg1, IDALsPrecSetupFn farg2, IDALsPrecSolveFn farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
IDALsPrecSetupFn arg2 = (IDALsPrecSetupFn) 0 ;
IDALsPrecSolveFn arg3 = (IDALsPrecSolveFn) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (IDALsPrecSetupFn)(farg2);
arg3 = (IDALsPrecSolveFn)(farg3);
result = (int)IDASetPreconditioner(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetJacTimes(void *farg1, IDALsJacTimesSetupFn farg2, IDALsJacTimesVecFn farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
IDALsJacTimesSetupFn arg2 = (IDALsJacTimesSetupFn) 0 ;
IDALsJacTimesVecFn arg3 = (IDALsJacTimesVecFn) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (IDALsJacTimesSetupFn)(farg2);
arg3 = (IDALsJacTimesVecFn)(farg3);
result = (int)IDASetJacTimes(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetEpsLin(void *farg1, double const *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype arg2 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype)(*farg2);
result = (int)IDASetEpsLin(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetLSNormFactor(void *farg1, double const *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype arg2 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype)(*farg2);
result = (int)IDASetLSNormFactor(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetLinearSolutionScaling(void *farg1, int const *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
result = (int)IDASetLinearSolutionScaling(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetIncrementFactor(void *farg1, double const *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype arg2 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype)(*farg2);
result = (int)IDASetIncrementFactor(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetLinWorkSpace(void *farg1, long *farg2, long *farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
long *arg3 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
arg3 = (long *)(farg3);
result = (int)IDAGetLinWorkSpace(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetNumJacEvals(void *farg1, long *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
result = (int)IDAGetNumJacEvals(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetNumPrecEvals(void *farg1, long *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
result = (int)IDAGetNumPrecEvals(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetNumPrecSolves(void *farg1, long *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
result = (int)IDAGetNumPrecSolves(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetNumLinIters(void *farg1, long *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
result = (int)IDAGetNumLinIters(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetNumLinConvFails(void *farg1, long *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
result = (int)IDAGetNumLinConvFails(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetNumJTSetupEvals(void *farg1, long *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
result = (int)IDAGetNumJTSetupEvals(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetNumJtimesEvals(void *farg1, long *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
result = (int)IDAGetNumJtimesEvals(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetNumLinResEvals(void *farg1, long *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
result = (int)IDAGetNumLinResEvals(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDAGetLastLinFlag(void *farg1, long *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
result = (int)IDAGetLastLinFlag(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT SwigArrayWrapper _wrap_FIDAGetLinReturnFlagName(long const *farg1) {
SwigArrayWrapper fresult ;
long arg1 ;
char *result = 0 ;

arg1 = (long)(*farg1);
result = (char *)IDAGetLinReturnFlagName(arg1);
fresult.size = strlen((const char*)(result));
fresult.data = (char *)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetLinearSolverB(void *farg1, int const *farg2, SUNLinearSolver farg3, SUNMatrix farg4) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
SUNLinearSolver arg3 = (SUNLinearSolver) 0 ;
SUNMatrix arg4 = (SUNMatrix) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
arg3 = (SUNLinearSolver)(farg3);
arg4 = (SUNMatrix)(farg4);
result = (int)IDASetLinearSolverB(arg1,arg2,arg3,arg4);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetJacFnB(void *farg1, int const *farg2, IDALsJacFnB farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
IDALsJacFnB arg3 = (IDALsJacFnB) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
arg3 = (IDALsJacFnB)(farg3);
result = (int)IDASetJacFnB(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetJacFnBS(void *farg1, int const *farg2, IDALsJacFnBS farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
IDALsJacFnBS arg3 = (IDALsJacFnBS) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
arg3 = (IDALsJacFnBS)(farg3);
result = (int)IDASetJacFnBS(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetEpsLinB(void *farg1, int const *farg2, double const *farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
realtype arg3 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
arg3 = (realtype)(*farg3);
result = (int)IDASetEpsLinB(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetLSNormFactorB(void *farg1, int const *farg2, double const *farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
realtype arg3 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
arg3 = (realtype)(*farg3);
result = (int)IDASetLSNormFactorB(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetLinearSolutionScalingB(void *farg1, int const *farg2, int const *farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
int arg3 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
arg3 = (int)(*farg3);
result = (int)IDASetLinearSolutionScalingB(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetIncrementFactorB(void *farg1, int const *farg2, double const *farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
realtype arg3 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
arg3 = (realtype)(*farg3);
result = (int)IDASetIncrementFactorB(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetPreconditionerB(void *farg1, int const *farg2, IDALsPrecSetupFnB farg3, IDALsPrecSolveFnB farg4) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
IDALsPrecSetupFnB arg3 = (IDALsPrecSetupFnB) 0 ;
IDALsPrecSolveFnB arg4 = (IDALsPrecSolveFnB) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
arg3 = (IDALsPrecSetupFnB)(farg3);
arg4 = (IDALsPrecSolveFnB)(farg4);
result = (int)IDASetPreconditionerB(arg1,arg2,arg3,arg4);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetPreconditionerBS(void *farg1, int const *farg2, IDALsPrecSetupFnBS farg3, IDALsPrecSolveFnBS farg4) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
IDALsPrecSetupFnBS arg3 = (IDALsPrecSetupFnBS) 0 ;
IDALsPrecSolveFnBS arg4 = (IDALsPrecSolveFnBS) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
arg3 = (IDALsPrecSetupFnBS)(farg3);
arg4 = (IDALsPrecSolveFnBS)(farg4);
result = (int)IDASetPreconditionerBS(arg1,arg2,arg3,arg4);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetJacTimesB(void *farg1, int const *farg2, IDALsJacTimesSetupFnB farg3, IDALsJacTimesVecFnB farg4) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
IDALsJacTimesSetupFnB arg3 = (IDALsJacTimesSetupFnB) 0 ;
IDALsJacTimesVecFnB arg4 = (IDALsJacTimesVecFnB) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
arg3 = (IDALsJacTimesSetupFnB)(farg3);
arg4 = (IDALsJacTimesVecFnB)(farg4);
result = (int)IDASetJacTimesB(arg1,arg2,arg3,arg4);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FIDASetJacTimesBS(void *farg1, int const *farg2, IDALsJacTimesSetupFnBS farg3, IDALsJacTimesVecFnBS farg4) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
IDALsJacTimesSetupFnBS arg3 = (IDALsJacTimesSetupFnBS) 0 ;
IDALsJacTimesVecFnBS arg4 = (IDALsJacTimesVecFnBS) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
arg3 = (IDALsJacTimesSetupFnBS)(farg3);
arg4 = (IDALsJacTimesVecFnBS)(farg4);
result = (int)IDASetJacTimesBS(arg1,arg2,arg3,arg4);
fresult = (int)(result);
return fresult;
}



