#ifndef __GENSOLV_MAIN_H__
#define __GENSOLV_MAIN_H__
#include "fptype.h"
#ifdef SINGLE_PRECISION
#define GENSOLV_MAIN	sgensolv_main
#define OMPSS_LU		ompss_sgetrf
#define OMPSS_LULL		ompss_sgetrf_ll
#endif
#ifdef DOUBLE_PRECISION
#define GENSOLV_MAIN	dgensolv_main
#define OMPSS_LU		ompss_dgetrf
#define OMPSS_LULL		ompss_dgetrf_ll
#endif
#pragma omp task in(ipiv[0;n-1]) inout(A[0;m*n-1])
void ompss_laswp(int n, fp_t *A, int m, int k1, int k2, int *ipiv, int stride);
int sgensolv_main(int m, int n, int b, float *A, float *B); 
int dgensolv_main(int m, int n, int b, double *A, double *B); 
#endif 
