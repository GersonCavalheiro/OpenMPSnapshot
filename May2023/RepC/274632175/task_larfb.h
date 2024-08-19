#ifndef __TASK_LARFB_H__
#define __TASK_LARFB_H__
#define LIBBBLAS_EXPORT __attribute__((__visibility__("default")))
#ifdef SINGLE_PRECISION
#define TASK_LARFB 			task_slarfb
#else
#define TASK_LARFB			task_dlarfb
#endif
#pragma omp task in( [ldt]T ) inout( [ldc]C ) priority(p)
extern LIBBBLAS_EXPORT void task_slarfb( int t, int m, int n, int k, int skip, float * V, int ldv, float * T, int ldt, float * C, int ldc, int p);
#pragma omp task in( [ldt]T ) inout( [ldc]C ) priority(p)
extern LIBBBLAS_EXPORT void task_dlarfb( int t, int m, int n, int k, int skip, double * V, int ldv, double * T, int ldt, double * C, int ldc, int p);
#endif 
