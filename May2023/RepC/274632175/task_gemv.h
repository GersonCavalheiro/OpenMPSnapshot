#ifndef __TASK_GEMV_H__
#define __TASK_GEMV_H__
#pragma omp task in([bm]A, [bm]X) inout([bm]Y) no_copy_deps                                                                                                                           
void task_sgemv(int bm, int bn, int m, int n, float alpha, float *A, float *X, float beta, float *Y);  
#pragma omp task in([bm]A, [bm]X) inout([bm]Y) no_copy_deps                                                                                                                           
void task_dgemv(int bm, int bn, int m, int n, double alpha, double *A, double *X, double beta, double *Y);  
#endif 
