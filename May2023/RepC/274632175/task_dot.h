#ifndef __TASK_DOT_H__
#define __TASK_DOT_H__
#include "async_struct.h"
#define LIBBBLAS_EXPORT __attribute__((__visibility__("default")))
#pragma omp task in([bm]X, [bm]Y) concurrent([bn]result) no_copy_deps priority(9999) label(ddot)
extern LIBBBLAS_EXPORT void task_ddot(int id, int idx, async_t *sync, int bm, int bn, int m, int n, double *X, double *Y, double *result);
#pragma omp task in([bm]X, [bm]Y) concurrent([bn]result) no_copy_deps priority(p) label(ddot_pure)
extern LIBBBLAS_EXPORT void task_ddot_pure(int p, int bm, int bn, int m, int n, double *X, double *Y, double *result);
#pragma omp task in([bm]X, [bm]Y) concurrent([bn]r1, [bn]r2) no_copy_deps priority(9999)
extern LIBBBLAS_EXPORT void task_ddot3(int id, int idx, async_t *sync, int bm, int bn, int m, int n, double *X, double *Y, double *r1, double *r2);
#pragma omp task in([bm]X, [bm]Y) concurrent([bn]result) no_copy_deps priority(9999) label(sdot)
extern LIBBBLAS_EXPORT void task_sdot(int id, int idx, async_t *sync, int bm, int bn, int m, int n, float *X, float *Y, float *result);
#pragma omp task in([bm]X, [bm]Y) concurrent([bn]result) no_copy_deps priority(p) label(sdot_pure)
extern LIBBBLAS_EXPORT void task_sdot_pure(int p, int bm, int bn, int m, int n, float *X, float *Y, float *result);
#pragma omp task in([bm]X, [bm]Y) concurrent([bn]r1, [bn]r2) no_copy_deps priority(9999)
extern LIBBBLAS_EXPORT void task_sdot3(int id, int idx, async_t *sync, int bm, int bn, int m, int n, float *X, float *Y, float *r1, float *r2);
#pragma omp task in([bn]prevr1, [bn]prevalpha, [bm]X, [bm]Y) concurrent([bn]r1, [bn]r2, [bn]alpha) no_copy_deps priority(9999)
extern LIBBBLAS_EXPORT void task_sdot4(int id, async_t *sync, int bm, int bn, int m, int n, float *prevr1, float *prevalpha, float *X, float *Y, float *r1, float *r2, float *alpha);
#pragma omp task in([bn]prevr1, [bn]prevalpha, [bm]X, [bm]Y) concurrent([bn]r1, [bn]r2, [bn]alpha) no_copy_deps priority(9999)
extern LIBBBLAS_EXPORT void task_ddot4(int id, async_t *sync, int bm, int bn, int m, int n, double *prevr1, double *prevalpha, double *X, double *Y, double *r1, double *r2, double *alpha);
#endif 
