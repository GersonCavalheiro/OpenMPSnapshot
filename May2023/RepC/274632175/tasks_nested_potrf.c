#include "tasks_nested_potrf.h"
#include "task_gemm.h"
#include "tasks_syrk.h"
#include "tasks_trsm.h"
#include "tasks_potrf.h"
#include "fptype.h"
#include "fpblas.h"
#include "fplapack.h"
#include "blas.h"
#include "matfprint.h"
#ifdef DOUBLE_PRECISION
#define __t_npotrf			ntask_dpotrf
#else
#define __t_npotrf			ntask_spotrf
#endif
#define DRYRUN	0
void __t_npotrf( int m, int b, int t, fp_t *A ) {
int k;
for (k = 0; k < m; k += b) {
TASK_POTRF(OMPSSLAPACK_LOWERTRIANG, b, &A[k*m+k], m, 1);
int i;
for (i = k + b; i < m; i+=b) {                     
TASK_TRSM(OMPSSBLAS_RIGHT, OMPSSBLAS_LOWERTRIANG, OMPSSBLAS_TRANSP, OMPSSBLAS_NDIAGUNIT, b, b, FP_ONE, &A[k*m+k], m, &A[k*m+i], m, 1);
}
for (i = k + b; i < m; i+=b) {                   
int j;
for (j = k + b; j < i; j+=b) {
TASK_GEMM(OMPSSBLAS_NTRANSP, OMPSSBLAS_TRANSP, b, b, b, FP_MONE, &A[k*m+i], m, &A[k*m+j], m, FP_ONE, &A[j*m+i], m, 1);
}
TASK_SYRK(OMPSSBLAS_LOWERTRIANG, OMPSSBLAS_NTRANSP, b, b, FP_MONE, &A[k*m+i], m, FP_ONE, &A[i*m+i], m, 1);
}
}
#pragma omp taskwait
}
