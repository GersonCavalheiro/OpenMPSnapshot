#ifndef __TASKS_SYRK_H__
#define __TASKS_SYRK_H__
#include "fpmatr.h"
#ifdef SINGLE_PRECISION		
#define TASK_SYRK		task_ssyrk
#else
#define TASK_SYRK		task_dsyrk
#endif
#define LIBBBLAS_EXPORT __attribute__((__visibility__("default")))
#pragma omp task in([n]A) inout([n]C) priority(p)
extern LIBBBLAS_EXPORT void task_ssyrk(ompssblas_t uplo, ompssblas_t transa, int n, int k, float alpha, float *A, int lda, float beta, float *C, int ldc, int p);
#pragma omp task in([n]A) inout([n]C) priority(p)
extern LIBBBLAS_EXPORT void task_dsyrk(ompssblas_t uplo, ompssblas_t transa, int n, int k, double alpha, double *A, int lda, double beta, double *C, int ldc, int p);
#endif 
