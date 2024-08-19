#ifndef __TASKS_SPOTRF_H__
#define __TASKS_SPOTRF_H__
#include "fplapack.h"
#include "fpmatr.h"
#ifdef SINGLE_PRECISION		
#define TASK_POTRF		task_spotrf
#else
#define TASK_POTRF		task_dpotrf
#endif
#define LIBBBLAS_EXPORT __attribute__((__visibility__("default")))
#pragma omp task inout([n]A) priority(p)
extern LIBBBLAS_EXPORT void task_spotrf(ompsslapack_t uplo, int n, float *A, int lda, int p);
#pragma omp task inout([n]A) priority(p)
extern LIBBBLAS_EXPORT void task_dpotrf(ompsslapack_t uplo, int n, double *A, int lda, int p);
#endif 
