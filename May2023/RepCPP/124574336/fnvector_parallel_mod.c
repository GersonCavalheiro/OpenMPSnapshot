






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


#include "nvector/nvector_parallel.h"

SWIGEXPORT N_Vector _wrap_FN_VNew_Parallel(int const *farg1, int64_t const *farg2, int64_t const *farg3, void *farg4) {
N_Vector fresult ;
MPI_Comm arg1 ;
sunindextype arg2 ;
sunindextype arg3 ;
SUNContext arg4 = (SUNContext) 0 ;
N_Vector result;

#ifdef SUNDIALS_MPI_ENABLED
arg1 = MPI_Comm_f2c((MPI_Fint)(*farg1));
#else
arg1 = *farg1;
#endif
arg2 = (sunindextype)(*farg2);
arg3 = (sunindextype)(*farg3);
arg4 = (SUNContext)(farg4);
result = (N_Vector)N_VNew_Parallel(arg1,arg2,arg3,arg4);
fresult = result;
return fresult;
}


SWIGEXPORT N_Vector _wrap_FN_VNewEmpty_Parallel(int const *farg1, int64_t const *farg2, int64_t const *farg3, void *farg4) {
N_Vector fresult ;
MPI_Comm arg1 ;
sunindextype arg2 ;
sunindextype arg3 ;
SUNContext arg4 = (SUNContext) 0 ;
N_Vector result;

#ifdef SUNDIALS_MPI_ENABLED
arg1 = MPI_Comm_f2c((MPI_Fint)(*farg1));
#else
arg1 = *farg1;
#endif
arg2 = (sunindextype)(*farg2);
arg3 = (sunindextype)(*farg3);
arg4 = (SUNContext)(farg4);
result = (N_Vector)N_VNewEmpty_Parallel(arg1,arg2,arg3,arg4);
fresult = result;
return fresult;
}


SWIGEXPORT N_Vector _wrap_FN_VMake_Parallel(int const *farg1, int64_t const *farg2, int64_t const *farg3, double *farg4, void *farg5) {
N_Vector fresult ;
MPI_Comm arg1 ;
sunindextype arg2 ;
sunindextype arg3 ;
realtype *arg4 = (realtype *) 0 ;
SUNContext arg5 = (SUNContext) 0 ;
N_Vector result;

#ifdef SUNDIALS_MPI_ENABLED
arg1 = MPI_Comm_f2c((MPI_Fint)(*farg1));
#else
arg1 = *farg1;
#endif
arg2 = (sunindextype)(*farg2);
arg3 = (sunindextype)(*farg3);
arg4 = (realtype *)(farg4);
arg5 = (SUNContext)(farg5);
result = (N_Vector)N_VMake_Parallel(arg1,arg2,arg3,arg4,arg5);
fresult = result;
return fresult;
}


SWIGEXPORT int64_t _wrap_FN_VGetLength_Parallel(N_Vector farg1) {
int64_t fresult ;
N_Vector arg1 = (N_Vector) 0 ;
sunindextype result;

arg1 = (N_Vector)(farg1);
result = N_VGetLength_Parallel(arg1);
fresult = (sunindextype)(result);
return fresult;
}


SWIGEXPORT int64_t _wrap_FN_VGetLocalLength_Parallel(N_Vector farg1) {
int64_t fresult ;
N_Vector arg1 = (N_Vector) 0 ;
sunindextype result;

arg1 = (N_Vector)(farg1);
result = N_VGetLocalLength_Parallel(arg1);
fresult = (sunindextype)(result);
return fresult;
}


SWIGEXPORT void _wrap_FN_VPrint_Parallel(N_Vector farg1) {
N_Vector arg1 = (N_Vector) 0 ;

arg1 = (N_Vector)(farg1);
N_VPrint_Parallel(arg1);
}


SWIGEXPORT void _wrap_FN_VPrintFile_Parallel(N_Vector farg1, void *farg2) {
N_Vector arg1 = (N_Vector) 0 ;
FILE *arg2 = (FILE *) 0 ;

arg1 = (N_Vector)(farg1);
arg2 = (FILE *)(farg2);
N_VPrintFile_Parallel(arg1,arg2);
}


SWIGEXPORT int _wrap_FN_VGetVectorID_Parallel(N_Vector farg1) {
int fresult ;
N_Vector arg1 = (N_Vector) 0 ;
N_Vector_ID result;

arg1 = (N_Vector)(farg1);
result = (N_Vector_ID)N_VGetVectorID_Parallel(arg1);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT N_Vector _wrap_FN_VCloneEmpty_Parallel(N_Vector farg1) {
N_Vector fresult ;
N_Vector arg1 = (N_Vector) 0 ;
N_Vector result;

arg1 = (N_Vector)(farg1);
result = (N_Vector)N_VCloneEmpty_Parallel(arg1);
fresult = result;
return fresult;
}


SWIGEXPORT N_Vector _wrap_FN_VClone_Parallel(N_Vector farg1) {
N_Vector fresult ;
N_Vector arg1 = (N_Vector) 0 ;
N_Vector result;

arg1 = (N_Vector)(farg1);
result = (N_Vector)N_VClone_Parallel(arg1);
fresult = result;
return fresult;
}


SWIGEXPORT void _wrap_FN_VDestroy_Parallel(N_Vector farg1) {
N_Vector arg1 = (N_Vector) 0 ;

arg1 = (N_Vector)(farg1);
N_VDestroy_Parallel(arg1);
}


SWIGEXPORT void _wrap_FN_VSpace_Parallel(N_Vector farg1, int64_t *farg2, int64_t *farg3) {
N_Vector arg1 = (N_Vector) 0 ;
sunindextype *arg2 = (sunindextype *) 0 ;
sunindextype *arg3 = (sunindextype *) 0 ;

arg1 = (N_Vector)(farg1);
arg2 = (sunindextype *)(farg2);
arg3 = (sunindextype *)(farg3);
N_VSpace_Parallel(arg1,arg2,arg3);
}


SWIGEXPORT double * _wrap_FN_VGetArrayPointer_Parallel(N_Vector farg1) {
double * fresult ;
N_Vector arg1 = (N_Vector) 0 ;
realtype *result = 0 ;

arg1 = (N_Vector)(farg1);
result = (realtype *)N_VGetArrayPointer_Parallel(arg1);
fresult = result;
return fresult;
}


SWIGEXPORT void _wrap_FN_VSetArrayPointer_Parallel(double *farg1, N_Vector farg2) {
realtype *arg1 = (realtype *) 0 ;
N_Vector arg2 = (N_Vector) 0 ;

arg1 = (realtype *)(farg1);
arg2 = (N_Vector)(farg2);
N_VSetArrayPointer_Parallel(arg1,arg2);
}


SWIGEXPORT void * _wrap_FN_VGetCommunicator_Parallel(N_Vector farg1) {
void * fresult ;
N_Vector arg1 = (N_Vector) 0 ;
void *result = 0 ;

arg1 = (N_Vector)(farg1);
result = (void *)N_VGetCommunicator_Parallel(arg1);
fresult = result;
return fresult;
}


SWIGEXPORT void _wrap_FN_VLinearSum_Parallel(double const *farg1, N_Vector farg2, double const *farg3, N_Vector farg4, N_Vector farg5) {
realtype arg1 ;
N_Vector arg2 = (N_Vector) 0 ;
realtype arg3 ;
N_Vector arg4 = (N_Vector) 0 ;
N_Vector arg5 = (N_Vector) 0 ;

arg1 = (realtype)(*farg1);
arg2 = (N_Vector)(farg2);
arg3 = (realtype)(*farg3);
arg4 = (N_Vector)(farg4);
arg5 = (N_Vector)(farg5);
N_VLinearSum_Parallel(arg1,arg2,arg3,arg4,arg5);
}


SWIGEXPORT void _wrap_FN_VConst_Parallel(double const *farg1, N_Vector farg2) {
realtype arg1 ;
N_Vector arg2 = (N_Vector) 0 ;

arg1 = (realtype)(*farg1);
arg2 = (N_Vector)(farg2);
N_VConst_Parallel(arg1,arg2);
}


SWIGEXPORT void _wrap_FN_VProd_Parallel(N_Vector farg1, N_Vector farg2, N_Vector farg3) {
N_Vector arg1 = (N_Vector) 0 ;
N_Vector arg2 = (N_Vector) 0 ;
N_Vector arg3 = (N_Vector) 0 ;

arg1 = (N_Vector)(farg1);
arg2 = (N_Vector)(farg2);
arg3 = (N_Vector)(farg3);
N_VProd_Parallel(arg1,arg2,arg3);
}


SWIGEXPORT void _wrap_FN_VDiv_Parallel(N_Vector farg1, N_Vector farg2, N_Vector farg3) {
N_Vector arg1 = (N_Vector) 0 ;
N_Vector arg2 = (N_Vector) 0 ;
N_Vector arg3 = (N_Vector) 0 ;

arg1 = (N_Vector)(farg1);
arg2 = (N_Vector)(farg2);
arg3 = (N_Vector)(farg3);
N_VDiv_Parallel(arg1,arg2,arg3);
}


SWIGEXPORT void _wrap_FN_VScale_Parallel(double const *farg1, N_Vector farg2, N_Vector farg3) {
realtype arg1 ;
N_Vector arg2 = (N_Vector) 0 ;
N_Vector arg3 = (N_Vector) 0 ;

arg1 = (realtype)(*farg1);
arg2 = (N_Vector)(farg2);
arg3 = (N_Vector)(farg3);
N_VScale_Parallel(arg1,arg2,arg3);
}


SWIGEXPORT void _wrap_FN_VAbs_Parallel(N_Vector farg1, N_Vector farg2) {
N_Vector arg1 = (N_Vector) 0 ;
N_Vector arg2 = (N_Vector) 0 ;

arg1 = (N_Vector)(farg1);
arg2 = (N_Vector)(farg2);
N_VAbs_Parallel(arg1,arg2);
}


SWIGEXPORT void _wrap_FN_VInv_Parallel(N_Vector farg1, N_Vector farg2) {
N_Vector arg1 = (N_Vector) 0 ;
N_Vector arg2 = (N_Vector) 0 ;

arg1 = (N_Vector)(farg1);
arg2 = (N_Vector)(farg2);
N_VInv_Parallel(arg1,arg2);
}


SWIGEXPORT void _wrap_FN_VAddConst_Parallel(N_Vector farg1, double const *farg2, N_Vector farg3) {
N_Vector arg1 = (N_Vector) 0 ;
realtype arg2 ;
N_Vector arg3 = (N_Vector) 0 ;

arg1 = (N_Vector)(farg1);
arg2 = (realtype)(*farg2);
arg3 = (N_Vector)(farg3);
N_VAddConst_Parallel(arg1,arg2,arg3);
}


SWIGEXPORT double _wrap_FN_VDotProd_Parallel(N_Vector farg1, N_Vector farg2) {
double fresult ;
N_Vector arg1 = (N_Vector) 0 ;
N_Vector arg2 = (N_Vector) 0 ;
realtype result;

arg1 = (N_Vector)(farg1);
arg2 = (N_Vector)(farg2);
result = (realtype)N_VDotProd_Parallel(arg1,arg2);
fresult = (realtype)(result);
return fresult;
}


SWIGEXPORT double _wrap_FN_VMaxNorm_Parallel(N_Vector farg1) {
double fresult ;
N_Vector arg1 = (N_Vector) 0 ;
realtype result;

arg1 = (N_Vector)(farg1);
result = (realtype)N_VMaxNorm_Parallel(arg1);
fresult = (realtype)(result);
return fresult;
}


SWIGEXPORT double _wrap_FN_VWrmsNorm_Parallel(N_Vector farg1, N_Vector farg2) {
double fresult ;
N_Vector arg1 = (N_Vector) 0 ;
N_Vector arg2 = (N_Vector) 0 ;
realtype result;

arg1 = (N_Vector)(farg1);
arg2 = (N_Vector)(farg2);
result = (realtype)N_VWrmsNorm_Parallel(arg1,arg2);
fresult = (realtype)(result);
return fresult;
}


SWIGEXPORT double _wrap_FN_VWrmsNormMask_Parallel(N_Vector farg1, N_Vector farg2, N_Vector farg3) {
double fresult ;
N_Vector arg1 = (N_Vector) 0 ;
N_Vector arg2 = (N_Vector) 0 ;
N_Vector arg3 = (N_Vector) 0 ;
realtype result;

arg1 = (N_Vector)(farg1);
arg2 = (N_Vector)(farg2);
arg3 = (N_Vector)(farg3);
result = (realtype)N_VWrmsNormMask_Parallel(arg1,arg2,arg3);
fresult = (realtype)(result);
return fresult;
}


SWIGEXPORT double _wrap_FN_VMin_Parallel(N_Vector farg1) {
double fresult ;
N_Vector arg1 = (N_Vector) 0 ;
realtype result;

arg1 = (N_Vector)(farg1);
result = (realtype)N_VMin_Parallel(arg1);
fresult = (realtype)(result);
return fresult;
}


SWIGEXPORT double _wrap_FN_VWL2Norm_Parallel(N_Vector farg1, N_Vector farg2) {
double fresult ;
N_Vector arg1 = (N_Vector) 0 ;
N_Vector arg2 = (N_Vector) 0 ;
realtype result;

arg1 = (N_Vector)(farg1);
arg2 = (N_Vector)(farg2);
result = (realtype)N_VWL2Norm_Parallel(arg1,arg2);
fresult = (realtype)(result);
return fresult;
}


SWIGEXPORT double _wrap_FN_VL1Norm_Parallel(N_Vector farg1) {
double fresult ;
N_Vector arg1 = (N_Vector) 0 ;
realtype result;

arg1 = (N_Vector)(farg1);
result = (realtype)N_VL1Norm_Parallel(arg1);
fresult = (realtype)(result);
return fresult;
}


SWIGEXPORT void _wrap_FN_VCompare_Parallel(double const *farg1, N_Vector farg2, N_Vector farg3) {
realtype arg1 ;
N_Vector arg2 = (N_Vector) 0 ;
N_Vector arg3 = (N_Vector) 0 ;

arg1 = (realtype)(*farg1);
arg2 = (N_Vector)(farg2);
arg3 = (N_Vector)(farg3);
N_VCompare_Parallel(arg1,arg2,arg3);
}


SWIGEXPORT int _wrap_FN_VInvTest_Parallel(N_Vector farg1, N_Vector farg2) {
int fresult ;
N_Vector arg1 = (N_Vector) 0 ;
N_Vector arg2 = (N_Vector) 0 ;
int result;

arg1 = (N_Vector)(farg1);
arg2 = (N_Vector)(farg2);
result = (int)N_VInvTest_Parallel(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FN_VConstrMask_Parallel(N_Vector farg1, N_Vector farg2, N_Vector farg3) {
int fresult ;
N_Vector arg1 = (N_Vector) 0 ;
N_Vector arg2 = (N_Vector) 0 ;
N_Vector arg3 = (N_Vector) 0 ;
int result;

arg1 = (N_Vector)(farg1);
arg2 = (N_Vector)(farg2);
arg3 = (N_Vector)(farg3);
result = (int)N_VConstrMask_Parallel(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT double _wrap_FN_VMinQuotient_Parallel(N_Vector farg1, N_Vector farg2) {
double fresult ;
N_Vector arg1 = (N_Vector) 0 ;
N_Vector arg2 = (N_Vector) 0 ;
realtype result;

arg1 = (N_Vector)(farg1);
arg2 = (N_Vector)(farg2);
result = (realtype)N_VMinQuotient_Parallel(arg1,arg2);
fresult = (realtype)(result);
return fresult;
}


SWIGEXPORT int _wrap_FN_VLinearCombination_Parallel(int const *farg1, double *farg2, void *farg3, N_Vector farg4) {
int fresult ;
int arg1 ;
realtype *arg2 = (realtype *) 0 ;
N_Vector *arg3 = (N_Vector *) 0 ;
N_Vector arg4 = (N_Vector) 0 ;
int result;

arg1 = (int)(*farg1);
arg2 = (realtype *)(farg2);
arg3 = (N_Vector *)(farg3);
arg4 = (N_Vector)(farg4);
result = (int)N_VLinearCombination_Parallel(arg1,arg2,arg3,arg4);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FN_VScaleAddMulti_Parallel(int const *farg1, double *farg2, N_Vector farg3, void *farg4, void *farg5) {
int fresult ;
int arg1 ;
realtype *arg2 = (realtype *) 0 ;
N_Vector arg3 = (N_Vector) 0 ;
N_Vector *arg4 = (N_Vector *) 0 ;
N_Vector *arg5 = (N_Vector *) 0 ;
int result;

arg1 = (int)(*farg1);
arg2 = (realtype *)(farg2);
arg3 = (N_Vector)(farg3);
arg4 = (N_Vector *)(farg4);
arg5 = (N_Vector *)(farg5);
result = (int)N_VScaleAddMulti_Parallel(arg1,arg2,arg3,arg4,arg5);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FN_VDotProdMulti_Parallel(int const *farg1, N_Vector farg2, void *farg3, double *farg4) {
int fresult ;
int arg1 ;
N_Vector arg2 = (N_Vector) 0 ;
N_Vector *arg3 = (N_Vector *) 0 ;
realtype *arg4 = (realtype *) 0 ;
int result;

arg1 = (int)(*farg1);
arg2 = (N_Vector)(farg2);
arg3 = (N_Vector *)(farg3);
arg4 = (realtype *)(farg4);
result = (int)N_VDotProdMulti_Parallel(arg1,arg2,arg3,arg4);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FN_VLinearSumVectorArray_Parallel(int const *farg1, double const *farg2, void *farg3, double const *farg4, void *farg5, void *farg6) {
int fresult ;
int arg1 ;
realtype arg2 ;
N_Vector *arg3 = (N_Vector *) 0 ;
realtype arg4 ;
N_Vector *arg5 = (N_Vector *) 0 ;
N_Vector *arg6 = (N_Vector *) 0 ;
int result;

arg1 = (int)(*farg1);
arg2 = (realtype)(*farg2);
arg3 = (N_Vector *)(farg3);
arg4 = (realtype)(*farg4);
arg5 = (N_Vector *)(farg5);
arg6 = (N_Vector *)(farg6);
result = (int)N_VLinearSumVectorArray_Parallel(arg1,arg2,arg3,arg4,arg5,arg6);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FN_VScaleVectorArray_Parallel(int const *farg1, double *farg2, void *farg3, void *farg4) {
int fresult ;
int arg1 ;
realtype *arg2 = (realtype *) 0 ;
N_Vector *arg3 = (N_Vector *) 0 ;
N_Vector *arg4 = (N_Vector *) 0 ;
int result;

arg1 = (int)(*farg1);
arg2 = (realtype *)(farg2);
arg3 = (N_Vector *)(farg3);
arg4 = (N_Vector *)(farg4);
result = (int)N_VScaleVectorArray_Parallel(arg1,arg2,arg3,arg4);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FN_VConstVectorArray_Parallel(int const *farg1, double const *farg2, void *farg3) {
int fresult ;
int arg1 ;
realtype arg2 ;
N_Vector *arg3 = (N_Vector *) 0 ;
int result;

arg1 = (int)(*farg1);
arg2 = (realtype)(*farg2);
arg3 = (N_Vector *)(farg3);
result = (int)N_VConstVectorArray_Parallel(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FN_VWrmsNormVectorArray_Parallel(int const *farg1, void *farg2, void *farg3, double *farg4) {
int fresult ;
int arg1 ;
N_Vector *arg2 = (N_Vector *) 0 ;
N_Vector *arg3 = (N_Vector *) 0 ;
realtype *arg4 = (realtype *) 0 ;
int result;

arg1 = (int)(*farg1);
arg2 = (N_Vector *)(farg2);
arg3 = (N_Vector *)(farg3);
arg4 = (realtype *)(farg4);
result = (int)N_VWrmsNormVectorArray_Parallel(arg1,arg2,arg3,arg4);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FN_VWrmsNormMaskVectorArray_Parallel(int const *farg1, void *farg2, void *farg3, N_Vector farg4, double *farg5) {
int fresult ;
int arg1 ;
N_Vector *arg2 = (N_Vector *) 0 ;
N_Vector *arg3 = (N_Vector *) 0 ;
N_Vector arg4 = (N_Vector) 0 ;
realtype *arg5 = (realtype *) 0 ;
int result;

arg1 = (int)(*farg1);
arg2 = (N_Vector *)(farg2);
arg3 = (N_Vector *)(farg3);
arg4 = (N_Vector)(farg4);
arg5 = (realtype *)(farg5);
result = (int)N_VWrmsNormMaskVectorArray_Parallel(arg1,arg2,arg3,arg4,arg5);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT double _wrap_FN_VDotProdLocal_Parallel(N_Vector farg1, N_Vector farg2) {
double fresult ;
N_Vector arg1 = (N_Vector) 0 ;
N_Vector arg2 = (N_Vector) 0 ;
realtype result;

arg1 = (N_Vector)(farg1);
arg2 = (N_Vector)(farg2);
result = (realtype)N_VDotProdLocal_Parallel(arg1,arg2);
fresult = (realtype)(result);
return fresult;
}


SWIGEXPORT double _wrap_FN_VMaxNormLocal_Parallel(N_Vector farg1) {
double fresult ;
N_Vector arg1 = (N_Vector) 0 ;
realtype result;

arg1 = (N_Vector)(farg1);
result = (realtype)N_VMaxNormLocal_Parallel(arg1);
fresult = (realtype)(result);
return fresult;
}


SWIGEXPORT double _wrap_FN_VMinLocal_Parallel(N_Vector farg1) {
double fresult ;
N_Vector arg1 = (N_Vector) 0 ;
realtype result;

arg1 = (N_Vector)(farg1);
result = (realtype)N_VMinLocal_Parallel(arg1);
fresult = (realtype)(result);
return fresult;
}


SWIGEXPORT double _wrap_FN_VL1NormLocal_Parallel(N_Vector farg1) {
double fresult ;
N_Vector arg1 = (N_Vector) 0 ;
realtype result;

arg1 = (N_Vector)(farg1);
result = (realtype)N_VL1NormLocal_Parallel(arg1);
fresult = (realtype)(result);
return fresult;
}


SWIGEXPORT double _wrap_FN_VWSqrSumLocal_Parallel(N_Vector farg1, N_Vector farg2) {
double fresult ;
N_Vector arg1 = (N_Vector) 0 ;
N_Vector arg2 = (N_Vector) 0 ;
realtype result;

arg1 = (N_Vector)(farg1);
arg2 = (N_Vector)(farg2);
result = (realtype)N_VWSqrSumLocal_Parallel(arg1,arg2);
fresult = (realtype)(result);
return fresult;
}


SWIGEXPORT double _wrap_FN_VWSqrSumMaskLocal_Parallel(N_Vector farg1, N_Vector farg2, N_Vector farg3) {
double fresult ;
N_Vector arg1 = (N_Vector) 0 ;
N_Vector arg2 = (N_Vector) 0 ;
N_Vector arg3 = (N_Vector) 0 ;
realtype result;

arg1 = (N_Vector)(farg1);
arg2 = (N_Vector)(farg2);
arg3 = (N_Vector)(farg3);
result = (realtype)N_VWSqrSumMaskLocal_Parallel(arg1,arg2,arg3);
fresult = (realtype)(result);
return fresult;
}


SWIGEXPORT int _wrap_FN_VInvTestLocal_Parallel(N_Vector farg1, N_Vector farg2) {
int fresult ;
N_Vector arg1 = (N_Vector) 0 ;
N_Vector arg2 = (N_Vector) 0 ;
int result;

arg1 = (N_Vector)(farg1);
arg2 = (N_Vector)(farg2);
result = (int)N_VInvTestLocal_Parallel(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FN_VConstrMaskLocal_Parallel(N_Vector farg1, N_Vector farg2, N_Vector farg3) {
int fresult ;
N_Vector arg1 = (N_Vector) 0 ;
N_Vector arg2 = (N_Vector) 0 ;
N_Vector arg3 = (N_Vector) 0 ;
int result;

arg1 = (N_Vector)(farg1);
arg2 = (N_Vector)(farg2);
arg3 = (N_Vector)(farg3);
result = (int)N_VConstrMaskLocal_Parallel(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT double _wrap_FN_VMinQuotientLocal_Parallel(N_Vector farg1, N_Vector farg2) {
double fresult ;
N_Vector arg1 = (N_Vector) 0 ;
N_Vector arg2 = (N_Vector) 0 ;
realtype result;

arg1 = (N_Vector)(farg1);
arg2 = (N_Vector)(farg2);
result = (realtype)N_VMinQuotientLocal_Parallel(arg1,arg2);
fresult = (realtype)(result);
return fresult;
}


SWIGEXPORT int _wrap_FN_VDotProdMultiLocal_Parallel(int const *farg1, N_Vector farg2, void *farg3, double *farg4) {
int fresult ;
int arg1 ;
N_Vector arg2 = (N_Vector) 0 ;
N_Vector *arg3 = (N_Vector *) 0 ;
realtype *arg4 = (realtype *) 0 ;
int result;

arg1 = (int)(*farg1);
arg2 = (N_Vector)(farg2);
arg3 = (N_Vector *)(farg3);
arg4 = (realtype *)(farg4);
result = (int)N_VDotProdMultiLocal_Parallel(arg1,arg2,arg3,arg4);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FN_VDotProdMultiAllReduce_Parallel(int const *farg1, N_Vector farg2, double *farg3) {
int fresult ;
int arg1 ;
N_Vector arg2 = (N_Vector) 0 ;
realtype *arg3 = (realtype *) 0 ;
int result;

arg1 = (int)(*farg1);
arg2 = (N_Vector)(farg2);
arg3 = (realtype *)(farg3);
result = (int)N_VDotProdMultiAllReduce_Parallel(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FN_VBufSize_Parallel(N_Vector farg1, int64_t *farg2) {
int fresult ;
N_Vector arg1 = (N_Vector) 0 ;
sunindextype *arg2 = (sunindextype *) 0 ;
int result;

arg1 = (N_Vector)(farg1);
arg2 = (sunindextype *)(farg2);
result = (int)N_VBufSize_Parallel(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FN_VBufPack_Parallel(N_Vector farg1, void *farg2) {
int fresult ;
N_Vector arg1 = (N_Vector) 0 ;
void *arg2 = (void *) 0 ;
int result;

arg1 = (N_Vector)(farg1);
arg2 = (void *)(farg2);
result = (int)N_VBufPack_Parallel(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FN_VBufUnpack_Parallel(N_Vector farg1, void *farg2) {
int fresult ;
N_Vector arg1 = (N_Vector) 0 ;
void *arg2 = (void *) 0 ;
int result;

arg1 = (N_Vector)(farg1);
arg2 = (void *)(farg2);
result = (int)N_VBufUnpack_Parallel(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FN_VEnableFusedOps_Parallel(N_Vector farg1, int const *farg2) {
int fresult ;
N_Vector arg1 = (N_Vector) 0 ;
int arg2 ;
int result;

arg1 = (N_Vector)(farg1);
arg2 = (int)(*farg2);
result = (int)N_VEnableFusedOps_Parallel(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FN_VEnableLinearCombination_Parallel(N_Vector farg1, int const *farg2) {
int fresult ;
N_Vector arg1 = (N_Vector) 0 ;
int arg2 ;
int result;

arg1 = (N_Vector)(farg1);
arg2 = (int)(*farg2);
result = (int)N_VEnableLinearCombination_Parallel(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FN_VEnableScaleAddMulti_Parallel(N_Vector farg1, int const *farg2) {
int fresult ;
N_Vector arg1 = (N_Vector) 0 ;
int arg2 ;
int result;

arg1 = (N_Vector)(farg1);
arg2 = (int)(*farg2);
result = (int)N_VEnableScaleAddMulti_Parallel(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FN_VEnableDotProdMulti_Parallel(N_Vector farg1, int const *farg2) {
int fresult ;
N_Vector arg1 = (N_Vector) 0 ;
int arg2 ;
int result;

arg1 = (N_Vector)(farg1);
arg2 = (int)(*farg2);
result = (int)N_VEnableDotProdMulti_Parallel(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FN_VEnableLinearSumVectorArray_Parallel(N_Vector farg1, int const *farg2) {
int fresult ;
N_Vector arg1 = (N_Vector) 0 ;
int arg2 ;
int result;

arg1 = (N_Vector)(farg1);
arg2 = (int)(*farg2);
result = (int)N_VEnableLinearSumVectorArray_Parallel(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FN_VEnableScaleVectorArray_Parallel(N_Vector farg1, int const *farg2) {
int fresult ;
N_Vector arg1 = (N_Vector) 0 ;
int arg2 ;
int result;

arg1 = (N_Vector)(farg1);
arg2 = (int)(*farg2);
result = (int)N_VEnableScaleVectorArray_Parallel(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FN_VEnableConstVectorArray_Parallel(N_Vector farg1, int const *farg2) {
int fresult ;
N_Vector arg1 = (N_Vector) 0 ;
int arg2 ;
int result;

arg1 = (N_Vector)(farg1);
arg2 = (int)(*farg2);
result = (int)N_VEnableConstVectorArray_Parallel(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FN_VEnableWrmsNormVectorArray_Parallel(N_Vector farg1, int const *farg2) {
int fresult ;
N_Vector arg1 = (N_Vector) 0 ;
int arg2 ;
int result;

arg1 = (N_Vector)(farg1);
arg2 = (int)(*farg2);
result = (int)N_VEnableWrmsNormVectorArray_Parallel(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FN_VEnableWrmsNormMaskVectorArray_Parallel(N_Vector farg1, int const *farg2) {
int fresult ;
N_Vector arg1 = (N_Vector) 0 ;
int arg2 ;
int result;

arg1 = (N_Vector)(farg1);
arg2 = (int)(*farg2);
result = (int)N_VEnableWrmsNormMaskVectorArray_Parallel(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FN_VEnableDotProdMultiLocal_Parallel(N_Vector farg1, int const *farg2) {
int fresult ;
N_Vector arg1 = (N_Vector) 0 ;
int arg2 ;
int result;

arg1 = (N_Vector)(farg1);
arg2 = (int)(*farg2);
result = (int)N_VEnableDotProdMultiLocal_Parallel(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT void * _wrap_FN_VCloneVectorArray_Parallel(int const *farg1, N_Vector farg2) {
void * fresult ;
int arg1 ;
N_Vector arg2 = (N_Vector) 0 ;
N_Vector *result = 0 ;

arg1 = (int)(*farg1);
arg2 = (N_Vector)(farg2);
result = (N_Vector *)N_VCloneVectorArray_Parallel(arg1,arg2);
fresult = result;
return fresult;
}


SWIGEXPORT void * _wrap_FN_VCloneVectorArrayEmpty_Parallel(int const *farg1, N_Vector farg2) {
void * fresult ;
int arg1 ;
N_Vector arg2 = (N_Vector) 0 ;
N_Vector *result = 0 ;

arg1 = (int)(*farg1);
arg2 = (N_Vector)(farg2);
result = (N_Vector *)N_VCloneVectorArrayEmpty_Parallel(arg1,arg2);
fresult = result;
return fresult;
}


SWIGEXPORT void _wrap_FN_VDestroyVectorArray_Parallel(void *farg1, int const *farg2) {
N_Vector *arg1 = (N_Vector *) 0 ;
int arg2 ;

arg1 = (N_Vector *)(farg1);
arg2 = (int)(*farg2);
N_VDestroyVectorArray_Parallel(arg1,arg2);
}



