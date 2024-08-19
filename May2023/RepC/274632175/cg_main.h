#ifndef __CG_MAIN_H__
#define __CG_MAIN_H__
#include "fpblas.h"
#include "fpsblas.h"
#include "bblas_copy.h"
#include "bblas_gemm.h"
#include "matfprint.h"
#include "bblas_convert.h"
#include "bblas_util.h"
#include "densutil.h"
#include "bblas_dot.h"
#include "bblas_dot_async.h" 
#include "bblas_gemm.h"
#include "bblas_axpy.h" 
#include "bblas_axpy_async.h" 
#include "dcsparse.h"
#include "hb.h"
#include "bsblas_csrmm.h"
#include "bsblas_csrmmb.h"
struct timeval start, stop;
static inline __attribute__((always_inline)) void dump_info(char *name, int k, double *residuals, unsigned int *elapse)
{
FILE *log = fopen(name, "w");
for ( int i = 0; i < k; i++ ) {
fprintf(log, "%d %E %u\n", i, residuals[i], elapse[i]);
}
fclose(log);
}
static inline __attribute__((always_inline)) void start_timer()
{
gettimeofday(&start, NULL);
}
static inline __attribute__((always_inline)) void stop_timer(unsigned int *elp)
{
gettimeofday(&stop, NULL);
*elp = (stop.tv_sec - start.tv_sec) * 1e6 + stop.tv_usec - start.tv_usec;
}
static inline __attribute__((always_inline)) void CG_DOT2(int p, int bm, int bn, int m, int n, double *X, double *Y, double *result, double *A, double *B, double *result2) 
{
int j;
for ( j=0; j<n; j+=bn ) {
int ds = n - j;
int d = ds < bn ? ds : bn;
int idx;
int i;
for ( i=0, idx=0; i<m; i+=bm, ++idx ) {
int cs = m - i;
int c = cs < bm ? cs : bm;
_cg_dot2(p, c, d, m, n, &X[j*m+i], &Y[j*m+i], result, &A[j*m+i], &B[j*m+i], result2);
}
result += bn;
result2 += bn;
}
}
#pragma omp task in([bm]X, [bm]Y, [bm]A, [bm]B) concurrent([bn]result, [bn]result2) no_copy_deps priority(p) label(cg_dot2)
void _cg_dot2(int p, int bm, int bn, int m, int n, double *X, double *Y, double *result, double *A, double *B, double *result2) 
{
fp_t local_result[bn];
for ( int j=0; j<bn; ++j ) {
local_result[j] = BLAS_dot(bm, X, i_one, Y, i_one);
X += m;
Y += m;
}
fp_t local_result2[bn];
int j;
for ( int j=0; j<bn; ++j ) {
local_result2[j] = BLAS_dot(bm, A, i_one, B, i_one);
A += m;
B += m;
}
#pragma omp critical
{
BLAS_axpy(bn, FP_ONE, local_result, i_one, result, i_one);
BLAS_axpy(bn, FP_ONE, local_result2, i_one, result2, i_one);
}
}
#endif 
