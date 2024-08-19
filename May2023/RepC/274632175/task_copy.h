#ifndef __TASK_COPY_H__
#define __TASK_COPY_H__
#include "selfsched.h"
#include "async_struct.h"
#define LIBBBLAS_EXPORT __attribute__((__visibility__("default")))
#define OMPSS_PRIOR_DFLT 1
#pragma omp task in([bm]X) out([bm]Y) priority(p) label(dcopy) no_copy_deps
extern LIBBBLAS_EXPORT void task_dcopy(int p, int bm, int bn, int m, int n, double *X, double *Y);  
#pragma omp task in([bm]X) out([bm]Y) priority(p) label(scopy) no_copy_deps
extern LIBBBLAS_EXPORT void task_scopy(int p, int bm, int bn, int m, int n, float *X, float *Y);  
#endif 
