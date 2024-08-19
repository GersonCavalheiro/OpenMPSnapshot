#ifndef __TASK_AXPY_ASYNC_H__
#define __TASK_AXPY_ASYNC_H__
#include "async_struct.h"
#define LIBBBLAS_EXPORT __attribute__((__visibility__("default")))
#pragma omp task in([bm]X, [bn]Anum, [bn]Aden, [bn]D) inout([bm]Y) no_copy_deps priority(1) label(daxpy_async)
extern LIBBBLAS_EXPORT void task_daxpy_async(int dotid, async_t *sync, int bm, int bn, int m, int n, double *Anum, double *Aden, double *X, double *D, double *Y);
#pragma omp task in([bm]X, [bn]Anum, [bn]Aden, [bn]D, [bm]Y) out([bm]Z) no_copy_deps priority(1) label(dcpaxpy_async)
extern LIBBBLAS_EXPORT void task_dcpaxpy_async(int dotid, async_t *sync, int bm, int bn, int m, int n, double *Anum, double *Aden, double *X, double *D, double *Y, double *Z);
#pragma omp task in([bm]X, [bn]Anum, [bn]Aden) inout([bm]Y) no_copy_deps priority(1) label(scal_daxpy_async)
extern LIBBBLAS_EXPORT void task_scal_daxpy_async(int dotid, async_t *sync, int bm, int bn, int m, int n, double alpha, double *Anum, double *Aden, double *X, double *Y);
#pragma omp task in([bm]X, [bn]Anum, [bn]Aden, [bm]Y) out([bm]Z) no_copy_deps priority(1) label(scal_dcpaxpy_async)
extern LIBBBLAS_EXPORT void task_scal_dcpaxpy_async(int dotid, async_t *sync, int bm, int bn, int m, int n, double alpha, double *Anum, double *Aden, double *X, double *Y, double *Z);
#pragma omp task in([bm]X, [bn]Anum, [bn]Aden, [bn]D) inout([bm]Y) no_copy_deps priority(1) label(saxpy_async)
extern LIBBBLAS_EXPORT void task_saxpy_async(int dotid, async_t *sync, int bm, int bn, int m, int n, float *Anum, float *Aden, float *X, float *D, float *Y);
#pragma omp task in([bm]X, [bn]Anum, [bn]Aden, [bn]D, [bm]Y) out([bm]Z) no_copy_deps priority(1) label(scpaxpy_async)
extern LIBBBLAS_EXPORT void task_scpaxpy_async(int dotid, async_t *sync, int bm, int bn, int m, int n, float *Anum, float *Aden, float *X, float *D, float *Y, float *Z);
#pragma omp task in([bm]X, [bn]Anum, [bn]Aden) inout([bm]Y) no_copy_deps priority(1) label(scal_saxpy_async)
extern LIBBBLAS_EXPORT void task_scal_saxpy_async(int dotid, async_t *sync, int bm, int bn, int m, int n, float alpha, float *Anum, float *Aden, float *X, float *Y);
#pragma omp task in([bm]X, [bn]Anum, [bn]Aden, [bm]Y) out([bm]Z) no_copy_deps priority(1) label(scal_scpaxpy_async)
extern LIBBBLAS_EXPORT void task_scal_scpaxpy_async(int dotid, async_t *sync, int bm, int bn, int m, int n, float alpha, float *Anum, float *Aden, float *X, float *Y, float *Z);
#pragma omp task in([bm]X1, [bm]X2, [bn]Anum, [bn]Aden, [bm]Y1, [bm]Y2) out([bm]Z1, [bm]Z2) priority(p) label(scpaxpy_comb_async)
extern LIBBBLAS_EXPORT void task_scal_scpaxpy_comb_async(int dotid, int l, async_t *sync, int p, int bm, int bn, int m, int n, float alpha, float *Anum, float *Aden, float *X1, float *X2, float *Y1, float *Y2, float *Z1, float *Z2);
#pragma omp task in([bm]X1, [bm]X2, [bn]Anum, [bn]Aden, [bm]Y1, [bm]Y2) out([bm]Z1, [bm]Z2) priority(p) label(scal_dcpaxpy_comb_async)
extern LIBBBLAS_EXPORT void task_scal_dcpaxpy_comb_async(int dotid, int l, async_t *sync, int p, int bm, int bn, int m, int n, double alpha, double *Anum, double *Aden, double *X1, double *X2, double *Y1, double *Y2, double *Z1, double *Z2);
#pragma omp task in([bm]X1, [bm]X2, [bn]Anum, [bn]Aden, [bm]Y1, [bm]Y2) out([bm]Z1, [bm]Z2) priority(p)
extern LIBBBLAS_EXPORT void task_scal_scpaxpy_comb_async_release(int dotid, int l, async_t *sync, int p, int bm, int bn, int m, int n, float alpha, float *Anum, float *Aden, float *X1, float *X2, float *Y1, float *Y2, float *Z1, float *Z2);
#pragma omp task in([bm]X1, [bm]X2, [bn]Anum, [bn]Aden, [bm]Y1, [bm]Y2) out([bm]Z1, [bm]Z2) priority(p)
extern LIBBBLAS_EXPORT void task_scal_dcpaxpy_comb_async_release(int dotid, int l, async_t *sync, int p, int bm, int bn, int m, int n, double alpha, double *Anum, double *Aden, double *X1, double *X2, double *Y1, double *Y2, double *Z1, double *Z2);
#pragma omp task in([bm]X1, [bm]X2, [bn]Anum, [bn]Aden, [bm]Y1, [bm]Y2) out([bm]Z1, [bm]Z2) priority(p) no_copy_deps label(scal_scpaxpy_comb_async_con)
extern LIBBBLAS_EXPORT void task_scal_scpaxpy_comb_async_concurrent(int dotid, int l, async_t *sync, int p, int bm, int bn, int m, int n, float alpha, float *Anum, float *Aden, float *X1, float *X2, float *Y1, float *Y2, float *Z1, float *Z2);
#pragma omp task in([bm]X1, [bm]X2, [bn]Anum, [bn]Aden, [bm]Y1, [bm]Y2) out([bm]Z1, [bm]Z2) priority(p) no_copy_deps label(scal_dcpaxpy_comb_async_con)
extern LIBBBLAS_EXPORT void task_scal_dcpaxpy_comb_async_concurrent(int dotid, int l, async_t *sync, int p, int bm, int bn, int m, int n, double alpha, double *Anum, double *Aden, double *X1, double *X2, double *Y1, double *Y2, double *Z1, double *Z2);
#pragma omp task in([bn]gamma2, [bn]sigma2, [bm]P2, [bm]V2, [bm]X2, [bm]R2, [bm]S) out([bm]P1, [bm]V1, [bm]X1, [bm]R1) priority(p)
extern LIBBBLAS_EXPORT void task_scpaxpy_comb4_async(int dotid, async_t *sync, int p, int bm, int bn, int m, int n, float *gamma1, float *gamma2, float *delta, float *sigma2, float *P2, float *V2, float *X2, float *R2, float *S, float *sigma1, float *P1, float *V1, float *X1, float *R1);
#pragma omp task in([bn]gamma2, [bn]sigma2, [bm]P2, [bm]V2, [bm]X2, [bm]R2, [bm]S) out([bm]P1, [bm]V1, [bm]X1, [bm]R1) priority(p)
extern LIBBBLAS_EXPORT void task_dcpaxpy_comb4_async(int dotid, async_t *sync, int p, int bm, int bn, int m, int n, double *gamma1, double *gamma2, double *delta, double *sigma2, double *P2, double *V2, double *X2, double *R2, double *S, double *sigma1, double *P1, double *V1, double *X1, double *R1);
#endif 
