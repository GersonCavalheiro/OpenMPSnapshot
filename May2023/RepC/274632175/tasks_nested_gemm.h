#ifndef __NTASKS_GEMM_H__
#define __NTASKS_GEMM_H__
#include "fpblas.h"
#define LIBBBLAS_EXPORT __attribute__((__visibility__("default")))
#ifdef SINGLE_PRECISION
#define NTASK_GEMM 			ntask_sgemm
#else
#define NTASK_GEMM 			ntask_dgemm
#endif
#pragma omp target device (smp)
#pragma omp task in ([m*m]A, [m*m]B) inout ([m*m]C)
extern LIBBBLAS_EXPORT void ntask_dgemm(int m, int b, int t, double *A, double *B, double *C);
#pragma omp target device (smp)
#pragma omp task in ([m*m]A, [m*m]B) inout ([m*m]C)
extern LIBBBLAS_EXPORT void ntask_sgemm(int m, int b, int t, float *A, float *B, float *C);
#endif 
