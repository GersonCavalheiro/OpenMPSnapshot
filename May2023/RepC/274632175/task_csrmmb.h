#ifndef __TASK_CSRMMB_H__
#define __TASK_CSRMMB_H__
#include "hb.h"
#include "selfsched.h"
#ifdef SINGLE_PRECISION
#define TASK_CSRMMB		task_scsrmmb
#else
#define TASK_CSRMMB		task_scsrmmb
#endif
#define LIBSBBLAS_EXPORT __attribute__((__visibility__("default")))
#pragma omp task in([A->n * n]B) inout([A->m * n]C)
extern LIBSBBLAS_EXPORT void task_scsrmmb(int n, float alpha, char *descra, hbmat_t *A, float *B, int ldb, float beta, float *C, int ldc, int si, selfsched_t *sched);
#pragma omp task in([A->n * n]B) inout([A->m * n]C)
extern LIBSBBLAS_EXPORT void task_dcsrmmb(int n, double alpha, char *descra, hbmat_t *A, double *B, int ldb, double beta, double *C, int ldc, int si, selfsched_t *sched);
#endif 
