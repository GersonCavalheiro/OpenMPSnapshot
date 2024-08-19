#ifndef __TASKS_GEMV_CSR_H__
#define __TASKS_GEMV_CSR_H__
#include "fpblas.h"
#include "fpsblas.h"
#include "fptype.h"
#include "blas.h"
#include "hb.h"
#define LIBSBBLAS_EXPORT __attribute__((__visibility__("default")))
#ifdef SINGLE_PRECISION
#define TASK_GEMV_CSR 		task_sgemv_csr
#define TASK_CPGEMV_CSR		task_scpgemv_csr
#else
#define TASK_GEMV_CSR		task_dgemv_csr
#define TASK_CPGEMV_CSR		task_dcpgemv_csr
#endif
#pragma omp task in(X[0;A->n]) inout(Y[0;A->m])
extern LIBSBBLAS_EXPORT void task_dgemv_csr(hbmat_t *A, char *trans, fp_t alpha, char *matdescra, fp_t beta, fp_t *X, fp_t *Y);
#pragma omp task in(X[0;A->n]) inout(Y[0;A->m])
extern LIBSBBLAS_EXPORT void task_sgemv_csr(hbmat_t *A, char *trans, fp_t alpha, char *matdescra, fp_t beta, fp_t *X, fp_t *Y);
#pragma omp task in(X[0;A->m], B[0;A->m]) out(Y[0;A->m])
extern LIBSBBLAS_EXPORT void task_dcpgemv_csr(hbmat_t *A, char *trans, fp_t alpha, char *matdescra, fp_t beta, fp_t *X, fp_t *B, fp_t *Y);
#pragma omp task in(X[0;A->m], B[0;A->m]) out(Y[0;A->m])
extern LIBSBBLAS_EXPORT void task_scpgemv_csr(hbmat_t *A, char *trans, fp_t alpha, char *matdescra, fp_t beta, fp_t *X, fp_t *B, fp_t *Y);
#endif 
