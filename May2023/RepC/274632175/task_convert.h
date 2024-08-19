#ifndef __TASK_CONVERT_H__
#define __TASK_CONVERT_H__
#define LIBBBLAS_EXPORT __attribute__((__visibility__("default")))
#pragma omp task in([bm]D) out([bm]S) no_copy_deps label(d2s)
extern LIBBBLAS_EXPORT void task_d2s(int bm, int bn, int m, int n, double *D, float *S);  
#pragma omp task out([bm]D) in([bm]S) no_copy_deps label(s2d)
extern LIBBBLAS_EXPORT void task_s2d(int bm, int bn, int m, int n, float *S, double *D);  
#endif 
