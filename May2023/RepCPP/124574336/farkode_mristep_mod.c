






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
{ printf("In " DECL ": " MSG); assert(0); RETURNNULL; }


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


#include "arkode/arkode_mristep.h"


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


#include <stdlib.h>
#ifdef _MSC_VER
# ifndef strtoull
#  define strtoull _strtoui64
# endif
# ifndef strtoll
#  define strtoll _strtoi64
# endif
#endif


#include <string.h>


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

SWIGEXPORT void _wrap_MRIStepCouplingMem_nmat_set(SwigClassWrapper const *farg1, int const *farg2) {
struct MRIStepCouplingMem *arg1 = (struct MRIStepCouplingMem *) 0 ;
int arg2 ;

SWIG_check_mutable_nonnull(*farg1, "struct MRIStepCouplingMem *", "MRIStepCouplingMem", "MRIStepCouplingMem::nmat", return );
arg1 = (struct MRIStepCouplingMem *)(farg1->cptr);
arg2 = (int)(*farg2);
if (arg1) (arg1)->nmat = arg2;
}


SWIGEXPORT int _wrap_MRIStepCouplingMem_nmat_get(SwigClassWrapper const *farg1) {
int fresult ;
struct MRIStepCouplingMem *arg1 = (struct MRIStepCouplingMem *) 0 ;
int result;

SWIG_check_mutable_nonnull(*farg1, "struct MRIStepCouplingMem *", "MRIStepCouplingMem", "MRIStepCouplingMem::nmat", return 0);
arg1 = (struct MRIStepCouplingMem *)(farg1->cptr);
result = (int) ((arg1)->nmat);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT void _wrap_MRIStepCouplingMem_stages_set(SwigClassWrapper const *farg1, int const *farg2) {
struct MRIStepCouplingMem *arg1 = (struct MRIStepCouplingMem *) 0 ;
int arg2 ;

SWIG_check_mutable_nonnull(*farg1, "struct MRIStepCouplingMem *", "MRIStepCouplingMem", "MRIStepCouplingMem::stages", return );
arg1 = (struct MRIStepCouplingMem *)(farg1->cptr);
arg2 = (int)(*farg2);
if (arg1) (arg1)->stages = arg2;
}


SWIGEXPORT int _wrap_MRIStepCouplingMem_stages_get(SwigClassWrapper const *farg1) {
int fresult ;
struct MRIStepCouplingMem *arg1 = (struct MRIStepCouplingMem *) 0 ;
int result;

SWIG_check_mutable_nonnull(*farg1, "struct MRIStepCouplingMem *", "MRIStepCouplingMem", "MRIStepCouplingMem::stages", return 0);
arg1 = (struct MRIStepCouplingMem *)(farg1->cptr);
result = (int) ((arg1)->stages);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT void _wrap_MRIStepCouplingMem_q_set(SwigClassWrapper const *farg1, int const *farg2) {
struct MRIStepCouplingMem *arg1 = (struct MRIStepCouplingMem *) 0 ;
int arg2 ;

SWIG_check_mutable_nonnull(*farg1, "struct MRIStepCouplingMem *", "MRIStepCouplingMem", "MRIStepCouplingMem::q", return );
arg1 = (struct MRIStepCouplingMem *)(farg1->cptr);
arg2 = (int)(*farg2);
if (arg1) (arg1)->q = arg2;
}


SWIGEXPORT int _wrap_MRIStepCouplingMem_q_get(SwigClassWrapper const *farg1) {
int fresult ;
struct MRIStepCouplingMem *arg1 = (struct MRIStepCouplingMem *) 0 ;
int result;

SWIG_check_mutable_nonnull(*farg1, "struct MRIStepCouplingMem *", "MRIStepCouplingMem", "MRIStepCouplingMem::q", return 0);
arg1 = (struct MRIStepCouplingMem *)(farg1->cptr);
result = (int) ((arg1)->q);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT void _wrap_MRIStepCouplingMem_p_set(SwigClassWrapper const *farg1, int const *farg2) {
struct MRIStepCouplingMem *arg1 = (struct MRIStepCouplingMem *) 0 ;
int arg2 ;

SWIG_check_mutable_nonnull(*farg1, "struct MRIStepCouplingMem *", "MRIStepCouplingMem", "MRIStepCouplingMem::p", return );
arg1 = (struct MRIStepCouplingMem *)(farg1->cptr);
arg2 = (int)(*farg2);
if (arg1) (arg1)->p = arg2;
}


SWIGEXPORT int _wrap_MRIStepCouplingMem_p_get(SwigClassWrapper const *farg1) {
int fresult ;
struct MRIStepCouplingMem *arg1 = (struct MRIStepCouplingMem *) 0 ;
int result;

SWIG_check_mutable_nonnull(*farg1, "struct MRIStepCouplingMem *", "MRIStepCouplingMem", "MRIStepCouplingMem::p", return 0);
arg1 = (struct MRIStepCouplingMem *)(farg1->cptr);
result = (int) ((arg1)->p);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT void _wrap_MRIStepCouplingMem_G_set(SwigClassWrapper const *farg1, void *farg2) {
struct MRIStepCouplingMem *arg1 = (struct MRIStepCouplingMem *) 0 ;
realtype ***arg2 = (realtype ***) 0 ;

SWIG_check_mutable_nonnull(*farg1, "struct MRIStepCouplingMem *", "MRIStepCouplingMem", "MRIStepCouplingMem::G", return );
arg1 = (struct MRIStepCouplingMem *)(farg1->cptr);
arg2 = (realtype ***)(farg2);
if (arg1) (arg1)->G = arg2;
}


SWIGEXPORT void * _wrap_MRIStepCouplingMem_G_get(SwigClassWrapper const *farg1) {
void * fresult ;
struct MRIStepCouplingMem *arg1 = (struct MRIStepCouplingMem *) 0 ;
realtype ***result = 0 ;

SWIG_check_mutable_nonnull(*farg1, "struct MRIStepCouplingMem *", "MRIStepCouplingMem", "MRIStepCouplingMem::G", return 0);
arg1 = (struct MRIStepCouplingMem *)(farg1->cptr);
result = (realtype ***) ((arg1)->G);
fresult = result;
return fresult;
}


SWIGEXPORT void _wrap_MRIStepCouplingMem_c_set(SwigClassWrapper const *farg1, double *farg2) {
struct MRIStepCouplingMem *arg1 = (struct MRIStepCouplingMem *) 0 ;
realtype *arg2 = (realtype *) 0 ;

SWIG_check_mutable_nonnull(*farg1, "struct MRIStepCouplingMem *", "MRIStepCouplingMem", "MRIStepCouplingMem::c", return );
arg1 = (struct MRIStepCouplingMem *)(farg1->cptr);
arg2 = (realtype *)(farg2);
if (arg1) (arg1)->c = arg2;
}


SWIGEXPORT double * _wrap_MRIStepCouplingMem_c_get(SwigClassWrapper const *farg1) {
double * fresult ;
struct MRIStepCouplingMem *arg1 = (struct MRIStepCouplingMem *) 0 ;
realtype *result = 0 ;

SWIG_check_mutable_nonnull(*farg1, "struct MRIStepCouplingMem *", "MRIStepCouplingMem", "MRIStepCouplingMem::c", return 0);
arg1 = (struct MRIStepCouplingMem *)(farg1->cptr);
result = (realtype *) ((arg1)->c);
fresult = result;
return fresult;
}


SWIGEXPORT SwigClassWrapper _wrap_new_MRIStepCouplingMem() {
SwigClassWrapper fresult ;
struct MRIStepCouplingMem *result = 0 ;

result = (struct MRIStepCouplingMem *)calloc(1, sizeof(struct MRIStepCouplingMem));
fresult.cptr = result;
fresult.cmemflags = SWIG_MEM_RVALUE | (1 ? SWIG_MEM_OWN : 0);
return fresult;
}


SWIGEXPORT void _wrap_delete_MRIStepCouplingMem(SwigClassWrapper *farg1) {
struct MRIStepCouplingMem *arg1 = (struct MRIStepCouplingMem *) 0 ;

SWIG_check_mutable(*farg1, "struct MRIStepCouplingMem *", "MRIStepCouplingMem", "MRIStepCouplingMem::~MRIStepCouplingMem()", return );
arg1 = (struct MRIStepCouplingMem *)(farg1->cptr);
free((char *) arg1);
}


SWIGEXPORT void _wrap_MRIStepCouplingMem_op_assign__(SwigClassWrapper *farg1, SwigClassWrapper const *farg2) {
struct MRIStepCouplingMem *arg1 = (struct MRIStepCouplingMem *) 0 ;
struct MRIStepCouplingMem *arg2 = 0 ;

(void)sizeof(arg1);
(void)sizeof(arg2);
SWIG_assign(farg1, *farg2);

}


SWIGEXPORT SwigClassWrapper _wrap_FMRIStepCoupling_LoadTable(int const *farg1) {
SwigClassWrapper fresult ;
int arg1 ;
MRIStepCoupling result;

arg1 = (int)(*farg1);
result = (MRIStepCoupling)MRIStepCoupling_LoadTable(arg1);
fresult.cptr = result;
fresult.cmemflags = SWIG_MEM_RVALUE | (0 ? SWIG_MEM_OWN : 0);
return fresult;
}


SWIGEXPORT SwigClassWrapper _wrap_FMRIStepCoupling_Alloc(int const *farg1, int const *farg2) {
SwigClassWrapper fresult ;
int arg1 ;
int arg2 ;
MRIStepCoupling result;

arg1 = (int)(*farg1);
arg2 = (int)(*farg2);
result = (MRIStepCoupling)MRIStepCoupling_Alloc(arg1,arg2);
fresult.cptr = result;
fresult.cmemflags = SWIG_MEM_RVALUE | (0 ? SWIG_MEM_OWN : 0);
return fresult;
}


SWIGEXPORT SwigClassWrapper _wrap_FMRIStepCoupling_Create(int const *farg1, int const *farg2, int const *farg3, int const *farg4, double *farg5, double *farg6) {
SwigClassWrapper fresult ;
int arg1 ;
int arg2 ;
int arg3 ;
int arg4 ;
realtype *arg5 = (realtype *) 0 ;
realtype *arg6 = (realtype *) 0 ;
MRIStepCoupling result;

arg1 = (int)(*farg1);
arg2 = (int)(*farg2);
arg3 = (int)(*farg3);
arg4 = (int)(*farg4);
arg5 = (realtype *)(farg5);
arg6 = (realtype *)(farg6);
result = (MRIStepCoupling)MRIStepCoupling_Create(arg1,arg2,arg3,arg4,arg5,arg6);
fresult.cptr = result;
fresult.cmemflags = SWIG_MEM_RVALUE | (0 ? SWIG_MEM_OWN : 0);
return fresult;
}


SWIGEXPORT SwigClassWrapper _wrap_FMRIStepCoupling_MIStoMRI(void *farg1, int const *farg2, int const *farg3) {
SwigClassWrapper fresult ;
ARKodeButcherTable arg1 = (ARKodeButcherTable) 0 ;
int arg2 ;
int arg3 ;
MRIStepCoupling result;

arg1 = (ARKodeButcherTable)(farg1);
arg2 = (int)(*farg2);
arg3 = (int)(*farg3);
result = (MRIStepCoupling)MRIStepCoupling_MIStoMRI(arg1,arg2,arg3);
fresult.cptr = result;
fresult.cmemflags = SWIG_MEM_RVALUE | (0 ? SWIG_MEM_OWN : 0);
return fresult;
}


SWIGEXPORT SwigClassWrapper _wrap_FMRIStepCoupling_Copy(SwigClassWrapper const *farg1) {
SwigClassWrapper fresult ;
MRIStepCoupling arg1 = (MRIStepCoupling) 0 ;
MRIStepCoupling result;

SWIG_check_mutable(*farg1, "MRIStepCoupling", "MRIStepCouplingMem", "MRIStepCoupling_Copy(MRIStepCoupling)", return SwigClassWrapper_uninitialized());
arg1 = (MRIStepCoupling)(farg1->cptr);
result = (MRIStepCoupling)MRIStepCoupling_Copy(arg1);
fresult.cptr = result;
fresult.cmemflags = SWIG_MEM_RVALUE | (0 ? SWIG_MEM_OWN : 0);
return fresult;
}


SWIGEXPORT void _wrap_FMRIStepCoupling_Space(SwigClassWrapper const *farg1, int64_t *farg2, int64_t *farg3) {
MRIStepCoupling arg1 = (MRIStepCoupling) 0 ;
sunindextype *arg2 = (sunindextype *) 0 ;
sunindextype *arg3 = (sunindextype *) 0 ;

SWIG_check_mutable(*farg1, "MRIStepCoupling", "MRIStepCouplingMem", "MRIStepCoupling_Space(MRIStepCoupling,sunindextype *,sunindextype *)", return );
arg1 = (MRIStepCoupling)(farg1->cptr);
arg2 = (sunindextype *)(farg2);
arg3 = (sunindextype *)(farg3);
MRIStepCoupling_Space(arg1,arg2,arg3);
}


SWIGEXPORT void _wrap_FMRIStepCoupling_Free(SwigClassWrapper const *farg1) {
MRIStepCoupling arg1 = (MRIStepCoupling) 0 ;

SWIG_check_mutable(*farg1, "MRIStepCoupling", "MRIStepCouplingMem", "MRIStepCoupling_Free(MRIStepCoupling)", return );
arg1 = (MRIStepCoupling)(farg1->cptr);
MRIStepCoupling_Free(arg1);
}


SWIGEXPORT void _wrap_FMRIStepCoupling_Write(SwigClassWrapper const *farg1, void *farg2) {
MRIStepCoupling arg1 = (MRIStepCoupling) 0 ;
FILE *arg2 = (FILE *) 0 ;

SWIG_check_mutable(*farg1, "MRIStepCoupling", "MRIStepCouplingMem", "MRIStepCoupling_Write(MRIStepCoupling,FILE *)", return );
arg1 = (MRIStepCoupling)(farg1->cptr);
arg2 = (FILE *)(farg2);
MRIStepCoupling_Write(arg1,arg2);
}


SWIGEXPORT int _wrap_FMRIStepGetCurrentButcherTables(void *farg1, void *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
ARKodeButcherTable *arg2 = (ARKodeButcherTable *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (ARKodeButcherTable *)(farg2);
result = (int)MRIStepGetCurrentButcherTables(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepWriteButcher(void *farg1, void *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
FILE *arg2 = (FILE *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (FILE *)(farg2);
result = (int)MRIStepWriteButcher(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT void * _wrap_FMRIStepCreate(ARKRhsFn farg1, double const *farg2, N_Vector farg3, int const *farg4, void *farg5) {
void * fresult ;
ARKRhsFn arg1 = (ARKRhsFn) 0 ;
realtype arg2 ;
N_Vector arg3 = (N_Vector) 0 ;
MRISTEP_ID arg4 ;
void *arg5 = (void *) 0 ;
void *result = 0 ;

arg1 = (ARKRhsFn)(farg1);
arg2 = (realtype)(*farg2);
arg3 = (N_Vector)(farg3);
arg4 = (MRISTEP_ID)(*farg4);
arg5 = (void *)(farg5);
result = (void *)MRIStepCreate(arg1,arg2,arg3,arg4,arg5);
fresult = result;
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepResize(void *farg1, N_Vector farg2, double const *farg3, ARKVecResizeFn farg4, void *farg5) {
int fresult ;
void *arg1 = (void *) 0 ;
N_Vector arg2 = (N_Vector) 0 ;
realtype arg3 ;
ARKVecResizeFn arg4 = (ARKVecResizeFn) 0 ;
void *arg5 = (void *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (N_Vector)(farg2);
arg3 = (realtype)(*farg3);
arg4 = (ARKVecResizeFn)(farg4);
arg5 = (void *)(farg5);
result = (int)MRIStepResize(arg1,arg2,arg3,arg4,arg5);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepReInit(void *farg1, ARKRhsFn farg2, double const *farg3, N_Vector farg4) {
int fresult ;
void *arg1 = (void *) 0 ;
ARKRhsFn arg2 = (ARKRhsFn) 0 ;
realtype arg3 ;
N_Vector arg4 = (N_Vector) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (ARKRhsFn)(farg2);
arg3 = (realtype)(*farg3);
arg4 = (N_Vector)(farg4);
result = (int)MRIStepReInit(arg1,arg2,arg3,arg4);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepReset(void *farg1, double const *farg2, N_Vector farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype arg2 ;
N_Vector arg3 = (N_Vector) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype)(*farg2);
arg3 = (N_Vector)(farg3);
result = (int)MRIStepReset(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepSStolerances(void *farg1, double const *farg2, double const *farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype arg2 ;
realtype arg3 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype)(*farg2);
arg3 = (realtype)(*farg3);
result = (int)MRIStepSStolerances(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepSVtolerances(void *farg1, double const *farg2, N_Vector farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype arg2 ;
N_Vector arg3 = (N_Vector) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype)(*farg2);
arg3 = (N_Vector)(farg3);
result = (int)MRIStepSVtolerances(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepWFtolerances(void *farg1, ARKEwtFn farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
ARKEwtFn arg2 = (ARKEwtFn) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (ARKEwtFn)(farg2);
result = (int)MRIStepWFtolerances(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepSetLinearSolver(void *farg1, SUNLinearSolver farg2, SUNMatrix farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
SUNLinearSolver arg2 = (SUNLinearSolver) 0 ;
SUNMatrix arg3 = (SUNMatrix) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (SUNLinearSolver)(farg2);
arg3 = (SUNMatrix)(farg3);
result = (int)MRIStepSetLinearSolver(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepRootInit(void *farg1, int const *farg2, ARKRootFn farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
ARKRootFn arg3 = (ARKRootFn) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
arg3 = (ARKRootFn)(farg3);
result = (int)MRIStepRootInit(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepSetDefaults(void *farg1) {
int fresult ;
void *arg1 = (void *) 0 ;
int result;

arg1 = (void *)(farg1);
result = (int)MRIStepSetDefaults(arg1);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepSetInterpolantType(void *farg1, int const *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
result = (int)MRIStepSetInterpolantType(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepSetInterpolantDegree(void *farg1, int const *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
result = (int)MRIStepSetInterpolantDegree(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepSetDenseOrder(void *farg1, int const *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
result = (int)MRIStepSetDenseOrder(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepSetNonlinearSolver(void *farg1, SUNNonlinearSolver farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
SUNNonlinearSolver arg2 = (SUNNonlinearSolver) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (SUNNonlinearSolver)(farg2);
result = (int)MRIStepSetNonlinearSolver(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepSetLinear(void *farg1, int const *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
result = (int)MRIStepSetLinear(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepSetNonlinear(void *farg1) {
int fresult ;
void *arg1 = (void *) 0 ;
int result;

arg1 = (void *)(farg1);
result = (int)MRIStepSetNonlinear(arg1);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepSetCoupling(void *farg1, SwigClassWrapper const *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
MRIStepCoupling arg2 = (MRIStepCoupling) 0 ;
int result;

arg1 = (void *)(farg1);
SWIG_check_mutable(*farg2, "MRIStepCoupling", "MRIStepCouplingMem", "MRIStepSetCoupling(void *,MRIStepCoupling)", return 0);
arg2 = (MRIStepCoupling)(farg2->cptr);
result = (int)MRIStepSetCoupling(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepSetTable(void *farg1, int const *farg2, void *farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
ARKodeButcherTable arg3 = (ARKodeButcherTable) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
arg3 = (ARKodeButcherTable)(farg3);
result = (int)MRIStepSetTable(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepSetTableNum(void *farg1, int const *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
result = (int)MRIStepSetTableNum(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepSetMaxNumSteps(void *farg1, long const *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
long arg2 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long)(*farg2);
result = (int)MRIStepSetMaxNumSteps(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepSetNonlinCRDown(void *farg1, double const *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype arg2 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype)(*farg2);
result = (int)MRIStepSetNonlinCRDown(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepSetNonlinRDiv(void *farg1, double const *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype arg2 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype)(*farg2);
result = (int)MRIStepSetNonlinRDiv(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepSetDeltaGammaMax(void *farg1, double const *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype arg2 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype)(*farg2);
result = (int)MRIStepSetDeltaGammaMax(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepSetLSetupFrequency(void *farg1, int const *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
result = (int)MRIStepSetLSetupFrequency(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepSetPredictorMethod(void *farg1, int const *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
result = (int)MRIStepSetPredictorMethod(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepSetMaxNonlinIters(void *farg1, int const *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
result = (int)MRIStepSetMaxNonlinIters(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepSetNonlinConvCoef(void *farg1, double const *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype arg2 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype)(*farg2);
result = (int)MRIStepSetNonlinConvCoef(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepSetMaxHnilWarns(void *farg1, int const *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
result = (int)MRIStepSetMaxHnilWarns(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepSetStopTime(void *farg1, double const *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype arg2 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype)(*farg2);
result = (int)MRIStepSetStopTime(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepSetFixedStep(void *farg1, double const *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype arg2 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype)(*farg2);
result = (int)MRIStepSetFixedStep(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepSetRootDirection(void *farg1, int *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
int *arg2 = (int *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int *)(farg2);
result = (int)MRIStepSetRootDirection(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepSetNoInactiveRootWarn(void *farg1) {
int fresult ;
void *arg1 = (void *) 0 ;
int result;

arg1 = (void *)(farg1);
result = (int)MRIStepSetNoInactiveRootWarn(arg1);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepSetErrHandlerFn(void *farg1, ARKErrHandlerFn farg2, void *farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
ARKErrHandlerFn arg2 = (ARKErrHandlerFn) 0 ;
void *arg3 = (void *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (ARKErrHandlerFn)(farg2);
arg3 = (void *)(farg3);
result = (int)MRIStepSetErrHandlerFn(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepSetErrFile(void *farg1, void *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
FILE *arg2 = (FILE *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (FILE *)(farg2);
result = (int)MRIStepSetErrFile(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepSetUserData(void *farg1, void *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
void *arg2 = (void *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (void *)(farg2);
result = (int)MRIStepSetUserData(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepSetDiagnostics(void *farg1, void *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
FILE *arg2 = (FILE *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (FILE *)(farg2);
result = (int)MRIStepSetDiagnostics(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepSetPostprocessStepFn(void *farg1, ARKPostProcessFn farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
ARKPostProcessFn arg2 = (ARKPostProcessFn) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (ARKPostProcessFn)(farg2);
result = (int)MRIStepSetPostprocessStepFn(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepSetPostprocessStageFn(void *farg1, ARKPostProcessFn farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
ARKPostProcessFn arg2 = (ARKPostProcessFn) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (ARKPostProcessFn)(farg2);
result = (int)MRIStepSetPostprocessStageFn(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepSetPreInnerFn(void *farg1, MRIStepPreInnerFn farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
MRIStepPreInnerFn arg2 = (MRIStepPreInnerFn) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (MRIStepPreInnerFn)(farg2);
result = (int)MRIStepSetPreInnerFn(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepSetPostInnerFn(void *farg1, MRIStepPostInnerFn farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
MRIStepPostInnerFn arg2 = (MRIStepPostInnerFn) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (MRIStepPostInnerFn)(farg2);
result = (int)MRIStepSetPostInnerFn(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepSetStagePredictFn(void *farg1, ARKStagePredictFn farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
ARKStagePredictFn arg2 = (ARKStagePredictFn) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (ARKStagePredictFn)(farg2);
result = (int)MRIStepSetStagePredictFn(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepSetJacFn(void *farg1, ARKLsJacFn farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
ARKLsJacFn arg2 = (ARKLsJacFn) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (ARKLsJacFn)(farg2);
result = (int)MRIStepSetJacFn(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepSetJacEvalFrequency(void *farg1, long const *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
long arg2 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long)(*farg2);
result = (int)MRIStepSetJacEvalFrequency(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepSetLinearSolutionScaling(void *farg1, int const *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
int arg2 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int)(*farg2);
result = (int)MRIStepSetLinearSolutionScaling(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepSetEpsLin(void *farg1, double const *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype arg2 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype)(*farg2);
result = (int)MRIStepSetEpsLin(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepSetLSNormFactor(void *farg1, double const *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype arg2 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype)(*farg2);
result = (int)MRIStepSetLSNormFactor(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepSetPreconditioner(void *farg1, ARKLsPrecSetupFn farg2, ARKLsPrecSolveFn farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
ARKLsPrecSetupFn arg2 = (ARKLsPrecSetupFn) 0 ;
ARKLsPrecSolveFn arg3 = (ARKLsPrecSolveFn) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (ARKLsPrecSetupFn)(farg2);
arg3 = (ARKLsPrecSolveFn)(farg3);
result = (int)MRIStepSetPreconditioner(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepSetJacTimes(void *farg1, ARKLsJacTimesSetupFn farg2, ARKLsJacTimesVecFn farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
ARKLsJacTimesSetupFn arg2 = (ARKLsJacTimesSetupFn) 0 ;
ARKLsJacTimesVecFn arg3 = (ARKLsJacTimesVecFn) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (ARKLsJacTimesSetupFn)(farg2);
arg3 = (ARKLsJacTimesVecFn)(farg3);
result = (int)MRIStepSetJacTimes(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepSetJacTimesRhsFn(void *farg1, ARKRhsFn farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
ARKRhsFn arg2 = (ARKRhsFn) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (ARKRhsFn)(farg2);
result = (int)MRIStepSetJacTimesRhsFn(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepSetLinSysFn(void *farg1, ARKLsLinSysFn farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
ARKLsLinSysFn arg2 = (ARKLsLinSysFn) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (ARKLsLinSysFn)(farg2);
result = (int)MRIStepSetLinSysFn(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepEvolve(void *farg1, double const *farg2, N_Vector farg3, double *farg4, int const *farg5) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype arg2 ;
N_Vector arg3 = (N_Vector) 0 ;
realtype *arg4 = (realtype *) 0 ;
int arg5 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype)(*farg2);
arg3 = (N_Vector)(farg3);
arg4 = (realtype *)(farg4);
arg5 = (int)(*farg5);
result = (int)MRIStepEvolve(arg1,arg2,arg3,arg4,arg5);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepGetDky(void *farg1, double const *farg2, int const *farg3, N_Vector farg4) {
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
result = (int)MRIStepGetDky(arg1,arg2,arg3,arg4);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepComputeState(void *farg1, N_Vector farg2, N_Vector farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
N_Vector arg2 = (N_Vector) 0 ;
N_Vector arg3 = (N_Vector) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (N_Vector)(farg2);
arg3 = (N_Vector)(farg3);
result = (int)MRIStepComputeState(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepGetNumRhsEvals(void *farg1, long *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
result = (int)MRIStepGetNumRhsEvals(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepGetNumLinSolvSetups(void *farg1, long *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
result = (int)MRIStepGetNumLinSolvSetups(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepGetCurrentCoupling(void *farg1, void *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
MRIStepCoupling *arg2 = (MRIStepCoupling *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (MRIStepCoupling *)(farg2);
result = (int)MRIStepGetCurrentCoupling(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepGetWorkSpace(void *farg1, long *farg2, long *farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
long *arg3 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
arg3 = (long *)(farg3);
result = (int)MRIStepGetWorkSpace(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepGetNumSteps(void *farg1, long *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
result = (int)MRIStepGetNumSteps(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepGetLastStep(void *farg1, double *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype *arg2 = (realtype *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype *)(farg2);
result = (int)MRIStepGetLastStep(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepGetCurrentTime(void *farg1, double *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype *arg2 = (realtype *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype *)(farg2);
result = (int)MRIStepGetCurrentTime(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepGetCurrentState(void *farg1, void *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
N_Vector *arg2 = (N_Vector *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (N_Vector *)(farg2);
result = (int)MRIStepGetCurrentState(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepGetCurrentGamma(void *farg1, double *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype *arg2 = (realtype *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype *)(farg2);
result = (int)MRIStepGetCurrentGamma(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepGetTolScaleFactor(void *farg1, double *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype *arg2 = (realtype *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype *)(farg2);
result = (int)MRIStepGetTolScaleFactor(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepGetErrWeights(void *farg1, N_Vector farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
N_Vector arg2 = (N_Vector) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (N_Vector)(farg2);
result = (int)MRIStepGetErrWeights(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepGetNumGEvals(void *farg1, long *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
result = (int)MRIStepGetNumGEvals(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepGetRootInfo(void *farg1, int *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
int *arg2 = (int *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int *)(farg2);
result = (int)MRIStepGetRootInfo(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepGetLastInnerStepFlag(void *farg1, int *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
int *arg2 = (int *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (int *)(farg2);
result = (int)MRIStepGetLastInnerStepFlag(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT SwigArrayWrapper _wrap_FMRIStepGetReturnFlagName(long const *farg1) {
SwigArrayWrapper fresult ;
long arg1 ;
char *result = 0 ;

arg1 = (long)(*farg1);
result = (char *)MRIStepGetReturnFlagName(arg1);
fresult.size = strlen((const char*)(result));
fresult.data = (char *)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepWriteParameters(void *farg1, void *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
FILE *arg2 = (FILE *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (FILE *)(farg2);
result = (int)MRIStepWriteParameters(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepWriteCoupling(void *farg1, void *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
FILE *arg2 = (FILE *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (FILE *)(farg2);
result = (int)MRIStepWriteCoupling(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepGetNonlinearSystemData(void *farg1, double *farg2, void *farg3, void *farg4, void *farg5, double *farg6, void *farg7, void *farg8) {
int fresult ;
void *arg1 = (void *) 0 ;
realtype *arg2 = (realtype *) 0 ;
N_Vector *arg3 = (N_Vector *) 0 ;
N_Vector *arg4 = (N_Vector *) 0 ;
N_Vector *arg5 = (N_Vector *) 0 ;
realtype *arg6 = (realtype *) 0 ;
N_Vector *arg7 = (N_Vector *) 0 ;
void **arg8 = (void **) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (realtype *)(farg2);
arg3 = (N_Vector *)(farg3);
arg4 = (N_Vector *)(farg4);
arg5 = (N_Vector *)(farg5);
arg6 = (realtype *)(farg6);
arg7 = (N_Vector *)(farg7);
arg8 = (void **)(farg8);
result = (int)MRIStepGetNonlinearSystemData(arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepGetNumNonlinSolvIters(void *farg1, long *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
result = (int)MRIStepGetNumNonlinSolvIters(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepGetNumNonlinSolvConvFails(void *farg1, long *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
result = (int)MRIStepGetNumNonlinSolvConvFails(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepGetNonlinSolvStats(void *farg1, long *farg2, long *farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
long *arg3 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
arg3 = (long *)(farg3);
result = (int)MRIStepGetNonlinSolvStats(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepGetLinWorkSpace(void *farg1, long *farg2, long *farg3) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
long *arg3 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
arg3 = (long *)(farg3);
result = (int)MRIStepGetLinWorkSpace(arg1,arg2,arg3);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepGetNumJacEvals(void *farg1, long *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
result = (int)MRIStepGetNumJacEvals(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepGetNumPrecEvals(void *farg1, long *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
result = (int)MRIStepGetNumPrecEvals(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepGetNumPrecSolves(void *farg1, long *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
result = (int)MRIStepGetNumPrecSolves(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepGetNumLinIters(void *farg1, long *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
result = (int)MRIStepGetNumLinIters(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepGetNumLinConvFails(void *farg1, long *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
result = (int)MRIStepGetNumLinConvFails(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepGetNumJTSetupEvals(void *farg1, long *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
result = (int)MRIStepGetNumJTSetupEvals(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepGetNumJtimesEvals(void *farg1, long *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
result = (int)MRIStepGetNumJtimesEvals(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepGetNumLinRhsEvals(void *farg1, long *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
result = (int)MRIStepGetNumLinRhsEvals(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT int _wrap_FMRIStepGetLastLinFlag(void *farg1, long *farg2) {
int fresult ;
void *arg1 = (void *) 0 ;
long *arg2 = (long *) 0 ;
int result;

arg1 = (void *)(farg1);
arg2 = (long *)(farg2);
result = (int)MRIStepGetLastLinFlag(arg1,arg2);
fresult = (int)(result);
return fresult;
}


SWIGEXPORT SwigArrayWrapper _wrap_FMRIStepGetLinReturnFlagName(long const *farg1) {
SwigArrayWrapper fresult ;
long arg1 ;
char *result = 0 ;

arg1 = (long)(*farg1);
result = (char *)MRIStepGetLinReturnFlagName(arg1);
fresult.size = strlen((const char*)(result));
fresult.data = (char *)(result);
return fresult;
}


SWIGEXPORT void _wrap_FMRIStepFree(void *farg1) {
void **arg1 = (void **) 0 ;

arg1 = (void **)(farg1);
MRIStepFree(arg1);
}


SWIGEXPORT void _wrap_FMRIStepPrintMem(void *farg1, void *farg2) {
void *arg1 = (void *) 0 ;
FILE *arg2 = (FILE *) 0 ;

arg1 = (void *)(farg1);
arg2 = (FILE *)(farg2);
MRIStepPrintMem(arg1,arg2);
}



