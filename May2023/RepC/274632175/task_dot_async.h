#ifndef __TASKS_DOT_ASYNC_H__
#define __TASKS_DOT_ASYNC_H__
#include "async_struct.h"
#define LIBBBLAS_EXPORT __attribute__((__visibility__("default")))
#pragma omp task in([bm]X, [bm]Y) no_copy_deps priority(p)
extern LIBBBLAS_EXPORT void task_sdot_async(int dotid, async_t *sync, int p, int bm, int bn, int m, int n, float *X, float *Y, float *result);
#pragma omp task in([bm]X, [bm]Y) no_copy_deps priority(p)
extern LIBBBLAS_EXPORT void task_ddot_async(int dotid, async_t *sync, int p, int bm, int bn, int m, int n, double *X, double *Y, double *result);
#pragma omp task in([bm]X, [bm]Y) no_copy_deps priority(p)
extern LIBBBLAS_EXPORT void task_sdot_sched_async(int dotid, int idx, async_t *sync, int p, int bm, int bn, int m, int n, float *X, float *Y, float *result, int release);
#pragma omp task in([bm]X, [bm]Y) no_copy_deps priority(p)
extern LIBBBLAS_EXPORT void task_ddot_sched_async(int dotid, int idx, async_t *sync, int p, int bm, int bn, int m, int n, double *X, double *Y, double *result, int release);
#pragma omp task in([bm]X, [bm]Y) no_copy_deps priority(p)
extern LIBBBLAS_EXPORT void task_sdot_sched_async_release(int dotid, int idx, async_t *sync, int p, int bm, int bn, int m, int n, float *X, float *Y, float *result, int release);
#pragma omp task in([bm]X, [bm]Y) no_copy_deps priority(p)
extern LIBBBLAS_EXPORT void task_ddot_sched_async_release(int dotid, int idx, async_t *sync, int p, int bm, int bn, int m, int n, double *X, double *Y, double *result, int release);
#pragma omp task in([bm]X, [bm]Y) concurrent([m]result) no_copy_deps priority(p) label(sdot_concurrent)
extern LIBBBLAS_EXPORT void task_sdot_sched_async_concurrent(int dotid, int idx, async_t *sync, int p, int bm, int bn, int m, int n, float *X, float *Y, float *result, int release, int *bitmap);
#pragma omp task in([bm]X, [bm]Y) concurrent([m]result) no_copy_deps priority(p) label(ddot_concurrent)
extern LIBBBLAS_EXPORT void task_ddot_sched_async_concurrent(int dotid, int idx, async_t *sync, int p, int bm, int bn, int m, int n, double *X, double *Y, double *result, int release, int *bitmap);
#pragma omp task in([bm]X, [bm]Y) concurrent([bn]r1) no_copy_deps priority(p) 
extern LIBBBLAS_EXPORT void task_sdot3_async(int id, int idx, async_t *sync, int p, int bm, int bn, int m, int n, float *X, float *Y, float *r1, float *r2, int release);
#pragma omp task in([bm]X, [bm]Y) concurrent([bn]r1) no_copy_deps priority(p)
extern LIBBBLAS_EXPORT void task_ddot3_async(int id, int idx, async_t *sync, int p, int bm, int bn, int m, int n, double *X, double *Y, double *r1, double *r2, int release);
#endif 
