#ifndef __NTASKS_POTRF_H__
#define __NTASKS_POTRF_H__
#include "fpblas.h"
#define LIBBBLAS_EXPORT __attribute__((__visibility__("default")))
#ifdef SINGLE_PRECISION
#define NTASK_POTRF 		ntask_spotrf
#else
#define NTASK_POTRF 		ntask_dpotrf
#endif
#pragma omp target device (smp)
#pragma omp task inout ([m*m]A) priority(1)
extern LIBBBLAS_EXPORT void ntask_dpotrf(int m, int b, int t, double *A );
#pragma omp target device (smp)
#pragma omp task inout ([m*m]A) priority(1)
extern LIBBBLAS_EXPORT void ntask_spotrf(int m, int b, int t, float *A );
#endif 
