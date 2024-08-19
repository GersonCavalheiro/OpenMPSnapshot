#ifndef __TASKS_TRSM_CSR_H__
#define __TASKS_TRSM_CSR_H__
#include "fptype.h"
#include "hb.h"
#define LIBSBBLAS_EXPORT __attribute__((__visibility__("default")))
#ifdef DOUBLE_PRECISION
#define TASK_TRSM_CSR 		task_dtrsm_csr
#else
#define TASK_TRSM_CSR		task_strsm_csr
#endif
#pragma omp task in(X[0;A->m]) out(Y[0;A->m])
extern LIBSBBLAS_EXPORT void task_dtrsm_csr(char *trans, fp_t alpha, char *matdescra, hbmat_t *A, fp_t *X, fp_t *Y);
#pragma omp task in(X[0;A->m]) out(Y[0;A->m])
extern LIBSBBLAS_EXPORT void task_strsm_csr(char *trans, fp_t alpha, char *matdescra, hbmat_t *A, fp_t *X, fp_t *Y);
#endif 
