#ifndef __TASK_TSQRT_H__
#define __TASK_TSQRT_H__
#define LIBBBLAS_EXPORT __attribute__((__visibility__("default")))
#ifdef SINGLE_PRECISION
#define TASK_TSQRT 			task_stsqrt
#else
#define TASK_TSQRT 			task_dtsqrt
#endif
#pragma omp task inout( [m*n]U, [m*n]D ) out( [nb_alg*m]S ) priority(p)
extern LIBBBLAS_EXPORT void task_stsqrt( int nb_alg, int m, int n, int skip, float *U, float *D, float *tau, float *S, int p);
#pragma omp task inout( [m*n]U, [m*n]D ) out( [nb_alg*m]S ) priority(p)
extern LIBBBLAS_EXPORT void task_dtsqrt( int nb_alg, int m, int n, int skip, double *U, double *D, double *tau, double *S, int p);
#endif 
