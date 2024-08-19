#ifndef __LU_KERNELS_H__
#define __LU_KERNELS_H__
#include "fptype.h"
#include "fpblas.h"
#include "fplapack.h"
#ifdef SINGLE_PRECISION
#define TASK__GETF2			task__sgetf2
#define TASK__LASWP			task__slaswp
#define TASK__TRSM_GEMM		task__strsm_sgemm
#define TASK_GETRF			task_sgetrf
#define TASK_LASWP			task_slaswp
#define TASK_TRSM_GEMM		task_strsm_sgemm
#define TASK_TRSM_GEMM_LL  	task_strsm_sgemm_ll
#pragma omp task inout(A[0;lda*n]) output(IPIV[0;n]) priority(1)
void task_sgetrf( int skip, int m, int n, float *A, int lda, int *IPIV);
#pragma omp task inout([lda*n]A) input([k2-k1]IPIV) priority(1)
void task_slaswp( int skip, int n, float *A, int lda, int k1, int k2, int *IPIV, int inc);
#pragma omp task input(A[0;dimR*dimC], ip[0;ips]) inout(B[0;dimC*dimM]) priority(1)
void task_strsm_sgemm(int skip, float *A, float *B, int dimM, int dimC, int dimR, int ips, int *ip, int lda);
#pragma omp task input(A[0;dimR*dimC], ip[0;ips]) inout(B[0;dimC*dimM]) priority(1)
void task_strsm_sgemm_ll(int skip, float *A, float *B, int dimM, int dimC, int dimR, int ips, int *ip, int lda);
#pragma omp task inout( A[0;m*n] ) output( IPIV[0;n] ) priority(3)
void task__sgetf2( int skip, int m, int n, float *A, int *IPIV);
#pragma omp task inout( A[0;m*n] ) input( IPIV[0;updw] ) priority(1)
void task__slaswp( int skip, int m, int n, int updw, float *A, int *IPIV);
#pragma omp task input( A[0;m*aw], IPIV[0;aw]) inout( B[0;m*n] ) priority(2)
void task__strsm_sgemm( int skip, int m, int n, int aw, int b, float *A, int *IPIV, float *B);
#endif
#ifdef DOUBLE_PRECISION
#define TASK__GETF2			task__dgetf2
#define TASK__LASWP			task__dlaswp
#define TASK__TRSM_GEMM		task__dtrsm_dgemm
#define TASK_GETRF			task_dgetrf
#define TASK_LASWP			task_dlaswp
#define TASK_TRSM_GEMM		task_dtrsm_dgemm
#define TASK_TRSM_GEMM_LL	task_dtrsm_dgemm_ll
#pragma omp task inout(A[0;lda*n]) output(IPIV[0;n]) priority(3)
void task_dgetrf( int skip, int m, int n, double *A, int lda, int *IPIV);
#pragma omp task inout([lda*n]A) input([k2-k1]IPIV) priority(1)
void task_dlaswp( int skip, int n, double *A, int lda, int k1, int k2, int *IPIV, int inc);
#pragma omp task input(A[0;dimR*dimC], ip[0;ips]) inout(B[0;dimC*dimM]) priority(1)
void task_dtrsm_dgemm(int skip, double *A, double *B, int dimM, int dimC, int dimR, int ips, int *ip, int lda);
#pragma omp task input(A[0;dimR*dimC], ip[0;ips]) inout(B[0;dimC*dimM]) priority(1)
void task_dtrsm_dgemm_ll(int skip, double *A, double *B, int dimM, int dimC, int dimR, int ips, int *ip, int lda);
#pragma omp task inout( A[0;m*n] ) output( IPIV[0;n] ) priority(3)
void task__dgetf2( int skip, int m, int n, double *A, int *IPIV);
#pragma omp task inout( A[0;m*n] ) input( IPIV[0;updw] ) priority(1)
void task__dlaswp( int skip, int m, int n, int updw, double *A, int *IPIV);
#pragma omp task input( A[0;m*aw], IPIV[0;aw]) inout( B[0;m*n] ) priority(2)
void task__dtrsm_dgemm( int skip, int m, int n, int aw, int b, double *A, int *IPIV, double *B);
#endif
#endif 
