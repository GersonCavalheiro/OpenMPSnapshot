#ifndef __TASKS_GEMM_H__
#define __TASKS_GEMM_H__
#include "fpmatr.h"
#include "async_struct.h"
#define LIBBBLAS_EXPORT __attribute__((__visibility__("default")))
#ifdef SINGLE_PRECISION
#define TASK_GEMM 			task_sgemm
#else
#define TASK_GEMM 			task_dgemm
#endif
#pragma omp task in([bm]A, [bk]B) inout([bm]C) no_copy_deps priority(p) label(dgemm)
extern LIBBBLAS_EXPORT void task_dgemm(ompssblas_t transa, ompssblas_t transb, int bm, int bn, int bk, double alpha, double *A, int ldimA, double *B, int ldimb, double beta, double *C, int ldimC, int p);
#pragma omp task in([bm]A, [bk]B) inout([bm]C) no_copy_deps priority(p) label(sgemm)
extern LIBBBLAS_EXPORT void task_sgemm(ompssblas_t transa, ompssblas_t transb, int bm, int bn, int bk, float alpha, float *A, int ldimA, float *B, int ldimB, float beta, float *C, int ldimC, int p);
#pragma omp task in([bm]A, [bk]B) inout([bm]C) no_copy_deps priority(p) label(sgemmcg)
extern LIBBBLAS_EXPORT void task_sgemmcg(int it, async_t *sync, ompssblas_t transa, ompssblas_t transb, int bm, int bn, int bk, float alpha, float *A, int ldimA, float *B, int ldimB, float beta, float *C, int ldimC, int p, int idx);
#pragma omp task in([bm]A, [bk]B) inout([bm]C) no_copy_deps priority(p) label(dgemmcg)
extern LIBBBLAS_EXPORT void task_dgemmcg(int it, async_t *sync, ompssblas_t transa, ompssblas_t transb, int bm, int bn, int bk, double alpha, double *A, int ldimA, double *B, int ldimB, double beta, double *C, int ldimC, int p, int idx);
#pragma omp task in([bm]A, [bk]B) inout([bm]C) no_copy_deps priority(p) label(sgemmcg_release)
extern LIBBBLAS_EXPORT void task_sgemmcg_release(int it, async_t *sync, ompssblas_t transa, ompssblas_t transb, int bm, int bn, int bk, float alpha, float *A, int ldimA, float *B, int ldimB, float beta, float *C, int ldimC, int p, int idx, int release, int *bitmap);
#pragma omp task in([bm]A, [bk]B) inout([bm]C) no_copy_deps priority(p) label(dgemmcg_release)
extern LIBBBLAS_EXPORT void task_dgemmcg_release(int it, async_t *sync, ompssblas_t transa, ompssblas_t transb, int bm, int bn, int bk, double alpha, double *A, int ldimA, double *B, int ldimB, double beta, double *C, int ldimC, int p, int idx, int release, int *bitmap);
#pragma omp task in([bm]A, [bk]B) inout([bm]C) no_copy_deps priority(p) label(sgemmcg_release)
extern LIBBBLAS_EXPORT void task_sgemmcg_switch(int it, async_t *sync, ompssblas_t transa, ompssblas_t transb, int bm, int bn, int bk, float alpha, float *A, int ldimA, float *B, int ldimB, float beta, float *C, int ldimC, int p, int idx, int release, int *bitmap, int on);
#pragma omp task in([bm]A, [bk]B) inout([bm]C) no_copy_deps priority(p) label(dgemmcg_release)
extern LIBBBLAS_EXPORT void task_dgemmcg_switch(int it, async_t *sync, ompssblas_t transa, ompssblas_t transb, int bm, int bn, int bk, double alpha, double *A, int ldimA, double *B, int ldimB, double beta, double *C, int ldimC, int p, int idx, int release, int *bitmap, int on);
#if 0
#pragma omp task in([bm]A, [bk]B) inout([bm]C) no_copy_deps priority(p)
extern LIBBBLAS_EXPORT void task_sgemm_sched(int bm, int bn, int bk, float alpha, float *A, int lda, int i, selfsched_t *schedB, float *B, int ldb, float beta, float *C, int ldc, int p); 
#pragma omp task in([bk]A, [bk]B) inout([bm]C) no_copy_deps priority(p)
extern LIBBBLAS_EXPORT void task_dgemm_sched(int bm, int bn, int bk, double alpha, double *A, int lda, int i, selfsched_t *schedB, double *B, int ldb, double beta, double *C, int ldc, int p); 
#pragma omp task in([bk]A, [bk]B) out([bm]C) no_copy_deps  priority(p)
extern LIBBBLAS_EXPORT void task_sgemm_prof(int it, async_t *sync, int idx, int p, int bm, int bn, int bk, int m, int n, int k, float alpha, float *A, int lda, selfsched_t *schedB, float *B, int ldb, float beta, float *C, int ldc);
#pragma omp task in([bk]A, [bk]B) out([bm]C) no_copy_deps  priority(p)
extern LIBBBLAS_EXPORT void task_dgemm_prof(int it, async_t *sync, int idx, int p, int bm, int bn, int bk, int m, int n, int k, double alpha, double *A, int lda, selfsched_t *schedB, double *B, int ldb, double beta, double *C, int ldc); 
#endif
#endif 
