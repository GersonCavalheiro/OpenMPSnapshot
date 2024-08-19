#ifndef __TASK_AXPY_H__
#define __TASK_AXPY_H__
#include "selfsched.h"
#define LIBBBLAS_EXPORT __attribute__((__visibility__("default")))
#pragma omp task in([bm]X, [bn]Anum, [bn]Aden) inout([bm]Y) no_copy_deps
extern LIBBBLAS_EXPORT void task_daxpy(int bm, int bn, int m, int n, double *Anum, double *Aden, double *X, double *Y);
#pragma omp task in([bm]X, [bn]Anum, [bn]Aden) inout([bm]Y) no_copy_deps
extern LIBBBLAS_EXPORT void task_scal_daxpy(int bm, int bn, int m, int n, double alpha, double *Anum, double *Aden, double *X, double *Y);
#pragma omp task in([bm]X, [bm]Y) out([bm]Z) no_copy_deps
extern LIBBBLAS_EXPORT void task_ext_daxpy(int bm, int bn, int m, int n, double SA, double *X, double *Y, double *Z);
#pragma omp task in([bm]X, bm[Y], [bn]SAnum, [bn]SAden) out([bm]Z) no_copy_deps priority(p)
extern LIBBBLAS_EXPORT void task_extm_daxpy(int bm, int bn, int m, int n, double *SAnum, double *SAden, double *X, double *Y, double *Z, int p);
#pragma omp task in([bm]X, [bn]Anum, [bn]Aden) inout([bm]Y) no_copy_deps
extern LIBBBLAS_EXPORT void task_saxpy(int bm, int bn, int m, int n, float *Anum, float *Aden, float *X, float *Y);
#pragma omp task in([bm]X, [bn]Anum, [bn]Aden) inout([bm]Y) no_copy_deps
extern LIBBBLAS_EXPORT void task_scal_saxpy(int bm, int bn, int m, int n, float alpha, float *Anum, float *Aden, float *X, float *Y);
#pragma omp task in([bm]X, [bm]Y) out([bm]Z) no_copy_deps
extern LIBBBLAS_EXPORT void task_ext_saxpy(int bm, int bn, int m, int n, float SA, float *X, float *Y, float *Z);
#pragma omp task in([bm]X, bm[Y], [bn]SAnum, [bn]SAden) out([bm]Z) no_copy_deps priority(p) label(extm_saxpy)
extern LIBBBLAS_EXPORT void task_extm_saxpy(int bm, int bn, int m, int n, float *SAnum, float *SAden, float *X, float *Y, float *Z, int p);
#pragma omp task in([bm]X, [bn]Anum, [bn]Aden) inout([bm]Y) out([bm]Z) no_copy_deps priority(1) label(extm_dcpaxpy)
extern LIBBBLAS_EXPORT void task_dcpaxpy(int bm, int bn, int m, int n, double *Anum, double *Aden, double *X, double *Y, double *Z);
#pragma omp task in([bm]X, [bn]Anum, [bn]Aden, [bm]Y) out([bm]Z) no_copy_deps priority(1)
extern LIBBBLAS_EXPORT void task_scal_dcpaxpy(int bm, int bn, int m, int n, double alpha, double *Anum, double *Aden, double *X, double *Y, double *Z);
#pragma omp task in([bm]X, [bn]Anum, [bn]Aden) inout([bm]Y) out([bm]Z) no_copy_deps priority(1)
extern LIBBBLAS_EXPORT void task_scpaxpy(int bm, int bn, int m, int n, float *Anum, float *Aden, float *X, float *Y, float *Z);
#pragma omp task in([bm]X, [bn]Anum, [bn]Aden, [bm]Y) out([bm]Z) no_copy_deps priority(1)
extern LIBBBLAS_EXPORT void task_scal_scpaxpy(int bm, int bn, int m, int n, float alpha, float *Anum, float *Aden, float *X, float *Y, float *Z);
#pragma omp task in([bm]X1, [bm]X2, [bn]Anum, [bn]Aden, [bm]Y1, [bm]Y2) out([bm]Z1, [bm]Z2) no_copy_deps priority(1) label(scpaxpy_comb)
extern LIBBBLAS_EXPORT void task_scpaxpy_comb(int bm, int bn, int m, int n, float alpha, float *Anum, float *Aden, float *X1, float *X2, float *Y1, float *Y2, float *Z1, float *Z2);
#pragma omp task in([bm]X1, [bm]X2, [bn]Anum, [bn]Aden, [bm]Y1, [bm]Y2) out([bm]Z1, [bm]Z2) no_copy_deps priority(1) label(dcpaxpy_comb)
extern LIBBBLAS_EXPORT void task_dcpaxpy_comb(int bm, int bn, int m, int n, double alpha, double *Anum, double *Aden, double *X1, double *X2, double *Y1, double *Y2, double *Z1, double *Z2);
#pragma omp task in([bn]gamma1, [bn]gamma2, [bn]delta, [bn]sigma2, [bm]P2, [bm]V2, [bm]X2, [bm]R2, [bm]S) out([bn]sigma1, [bm]P1, [bm]V1, [bm]X1, [bm]R1)
extern LIBBBLAS_EXPORT void task_scpaxpy_comb4(int bm, int bn, int m, int n, float *gamma1, float *gamma2, float *delta, float *sigma2, float *P2, float *V2, float *X2, float *R2, float *S,\
float *sigma1, float *P1, float *V1, float *X1, float *R1);
#pragma omp task in([bn]gamma1, [bn]gamma2, [bn]delta, [bn]sigma2, [bm]P2, [bm]V2, [bm]X2, [bm]R2, [bm]S) out([bn]sigma1, [bm]P1, [bm]V1, [bm]X1, [bm]R1)
extern LIBBBLAS_EXPORT void task_dcpaxpy_comb4(int bm, int bn, int m, int n, double *gamma1, double *gamma2, double *delta, double *sigma2, double *P2, double *V2, double *X2, double *R2, double *S,\
double *sigma1, double *P1, double *V1, double *X1, double *R1);
#pragma omp task in([bm]x) out([bm]y) no_copy_deps
extern LIBBBLAS_EXPORT void task_sdaxpy(int bm, int bn, int m, int n, float a, float *x, double *y);
#pragma omp task in([bm]x, [bm]y) out([bm]z) no_copy_deps label(sdcpaxpy)
extern LIBBBLAS_EXPORT void task_sdcpaxpy(int bm, int bn, int m, int n, double a, double *x, double *y, double *z);
#pragma omp task in([bn]gamman, [bn]gammad, [bn]alpha, [bm]xp, [bm]rp, [bm]q) inout([bm]z, [bm]s, [bm]p, [bm]w) out([bm]x, [bm]r) no_copy_deps
extern LIBBBLAS_EXPORT void task_saxpy4(int bm, int bn, int m, int n, float *gamman, float *gammad, float *alpha, float *z, float *s, float *p, float *q, float *w, float *xp, float *x, float *rp, float *r);
#pragma omp task in([bn]gamman, [bn]gammad, [bn]alpha, [bm]xp, [bm]rp, [bm]q) inout([bm]z, [bm]s, [bm]p, [bm]w) out([bm]x, [bm]r) no_copy_deps
extern LIBBBLAS_EXPORT void task_daxpy4(int bm, int bn, int m, int n, double *gamman, double *gammad, double *alpha, double *z, double *s, double *p, double *q, double *w, double *xp, double *x, double *rp, double *r);
#endif 
