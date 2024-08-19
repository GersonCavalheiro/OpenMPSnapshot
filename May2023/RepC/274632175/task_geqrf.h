#ifndef __TASK_GEQRF_H__
#define __TASK_GEQRF_H__
#define LIBBBLAS_EXPORT __attribute__((__visibility__("default")))
#ifdef SINGLE_PRECISION
#define TASK_GEQRF 			task_sgeqrf
#else
#define TASK_GEQRF			task_dgeqrf
#endif
#pragma omp task inout([m]A) out([n]tau, [n]S) priority(p)
extern LIBBBLAS_EXPORT void task_sgeqrf( int t, int m, int n, int skip, float * A, int lda, float * tau, float * S, int lds, int p); 
#pragma omp task inout([m]A) out([n]tau, [n]S) priority(p)
extern LIBBBLAS_EXPORT void task_dgeqrf( int t, int m, int n, int skip, double * A, int lda, double * tau, double * S, int lds, int p);
#endif 
