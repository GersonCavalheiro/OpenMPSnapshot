#ifndef __NTASKS_SYRK_H__
#define __NTASKS_SYRK_H__
#include "fpblas.h"
#define LIBBBLAS_EXPORT __attribute__((__visibility__("default")))
#ifdef SINGLE_PRECISION
#define NTASK_SYRK 			ntask_ssyrk
#else
#define NTASK_SYRK 			ntask_dsyrk
#endif
#pragma omp target device (smp)
#pragma omp task in ([m*m]A) inout ([m*m]C)
extern LIBBBLAS_EXPORT void ntask_dsyrk(int m, int b, int t, double *A, double *C);
#pragma omp target device (smp)
#pragma omp task in ([m*m]A) inout ([m*m]C)
extern LIBBBLAS_EXPORT void ntask_ssyrk(int m, int b, int t, float *A, float *C);
#endif 
