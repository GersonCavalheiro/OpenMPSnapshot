#ifndef __TASKS_TRSM_H__
#define __TASKS_TRSM_H__
#include "fpmatr.h"
#ifdef SINGLE_PRECISION
#define TASK_TRSM 			task_strsm
#endif
#ifdef DOUBLE_PRECISION
#define TASK_TRSM 			task_dtrsm
#endif
#define LIBBBLAS_EXPORT __attribute__((__visibility__("default")))
#pragma omp task in([bm*bm]A) inout([bm*bn]B) priority(p)
extern LIBBBLAS_EXPORT void task_strsm(ompssblas_t side, ompssblas_t uplo, ompssblas_t trans, ompssblas_t diag, int bm, int bn, float alpha, float *A, int lda, float *B, int ldb, int p);
#pragma omp task in([bm*bm]A) inout([bm*bn]B) priority(p)
extern LIBBBLAS_EXPORT void task_dtrsm(ompssblas_t side, ompssblas_t uplo, ompssblas_t trans, ompssblas_t diag, int bm, int bn, double alpha, double *A, int lda, double *B, int ldb, int p);
#endif 
