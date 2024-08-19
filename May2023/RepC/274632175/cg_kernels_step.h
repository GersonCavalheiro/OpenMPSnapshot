#ifndef __CG_KERNELS_STEP_H__
#define __CG_KERNELS_STEP_H__
#include "sparsematrix.h"
#pragma omp task in([nb*n]z2, [b]rhs) inout([b]z)
void stask_zred(int i, int b, int n, int nb, double *z2, double *z, double *rhs);
#endif 
