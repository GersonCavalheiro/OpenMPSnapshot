#ifndef __NTASKS_TRSM_H__
#define __NTASKS_TRSM_H__
#include "fpblas.h"
#define LIBBBLAS_EXPORT __attribute__((__visibility__("default")))
#ifdef SINGLE_PRECISION
#define NTASK_TRSM 			ntask_strsm
#else
#define NTASK_TRSM 			ntask_dtrsm
#endif
#pragma omp target device (smp)
#pragma omp task in ([m*m]A) inout ([m*m]B)
extern LIBBBLAS_EXPORT void ntask_dtrsm(int m, int b, int t, double *A, double *B);
#pragma omp target device (smp)
#pragma omp task in ([m*m]A) inout ([m*m]B)
extern LIBBBLAS_EXPORT void ntask_strsm(int m, int b, int t, float *A, float *B);
#endif 
