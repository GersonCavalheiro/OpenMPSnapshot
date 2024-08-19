#ifndef __CHOLS_KERNELS_H__
#define __CHOLS_KERNELS_H__
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <math.h>
#include "symfac.h"
#include "array.h"
#include "hb.h"
#include "fptype.h"
#ifdef SINGLE_PRECISION
#define POTRF_SPARSE_CSR		spotrf_sparse_csr	
#define SYRK_SPARSE_CSR			ssyrk_sparse_csr
#define GEMM_SPARSE_CSR			sgemm_sparse_csr
#define TRSM_SPARSE_CSR			strsm_sparse_csr
#define POTRF_SPARSE_CSC		spotrf_sparse_csc	
#define SYRK_SPARSE_CSC			ssyrk_sparse_csc
#define GEMM_SPARSE_CSC			sgemm_sparse_csc
#define TRSM_SPARSE_CSC			strsm_sparse_csc
#else
#define POTRF_SPARSE_CSR		dpotrf_sparse_csr	
#define SYRK_SPARSE_CSR			dsyrk_sparse_csr
#define GEMM_SPARSE_CSR			dgemm_sparse_csr
#define TRSM_SPARSE_CSR			dtrsm_sparse_csr
#define POTRF_SPARSE_CSC		dpotrf_sparse_csc	
#define SYRK_SPARSE_CSC			dsyrk_sparse_csc
#define GEMM_SPARSE_CSC			dgemm_sparse_csc
#define TRSM_SPARSE_CSC			dtrsm_sparse_csc
#endif
#pragma omp task inout([1]A) priority(1)
void spotrf_sparse_csc(hbmat_t* A);
#pragma omp task in([1]A) inout([1]B) priority(2)
void ssyrk_sparse_csc(hbmat_t* A, hbmat_t* B);
#pragma omp task in([1]A, [1]B) inout([1]C)
void sgemm_sparse_csc(hbmat_t* A, hbmat_t* B, hbmat_t* C);
#pragma omp task in([1]A) inout([1]B)
void strsm_sparse_csc(hbmat_t* A, hbmat_t* B);
#pragma omp task inout([1]A) priority(1)
void dpotrf_sparse_csc(hbmat_t* A);
#pragma omp task in([1]A) inout([1]B) priority(2)
void dsyrk_sparse_csc(hbmat_t* A, hbmat_t* B);
#pragma omp task in([1]A, [1]B) inout([1]C)
void dgemm_sparse_csc(hbmat_t* A, hbmat_t* B, hbmat_t* C);
#pragma omp task in([1]A) inout([1]B)
void dtrsm_sparse_csc(hbmat_t* A, hbmat_t* B);
#pragma omp task inout([1]A) priority(4)
void spotrf_sparse_csr(hbmat_t* A);
#pragma omp task in([1]A) inout([1]B) priority(3)
void ssyrk_sparse_csr(hbmat_t* A, hbmat_t* B);
#pragma omp task in([1]A, [1]B) inout([1]C)
void sgemm_sparse_csr(hbmat_t* A, hbmat_t* B, hbmat_t* C);
#pragma omp task in([1]A) inout([1]B)
void strsm_sparse_csr(hbmat_t* A, hbmat_t* B);
#pragma omp task inout([1]A) priority(4)
void dpotrf_sparse_csr(hbmat_t* A);
#pragma omp task in([1]A) inout([1]B) priority(3)
void dsyrk_sparse_csr(hbmat_t* A, hbmat_t* B);
#pragma omp task in([1]A, [1]B) inout([1]C)
void dgemm_sparse_csr(hbmat_t* A, hbmat_t* B, hbmat_t* C);
#pragma omp task in([1]A) inout([1]B)
void dtrsm_sparse_csr(hbmat_t* A, hbmat_t* B);
#endif
