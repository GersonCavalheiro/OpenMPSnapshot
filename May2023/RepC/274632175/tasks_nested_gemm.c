#include "tasks_nested_gemm.h"
#include "task_gemm.h"
#include "tasks_syrk.h"
#include "tasks_trsm.h"
#include "tasks_potrf.h"
#include "fptype.h"
#include "fpblas.h"
#include "blas.h"
#include "matfprint.h"
#ifdef DOUBLE_PRECISION
#define __t_ngemm			ntask_dgemm
#else
#define __t_ngemm			ntask_sgemm
#endif
#define DRYRUN	0
void __t_ngemm(int m, int b, int t, fp_t *A, fp_t *B, fp_t *C) {
int k;
for ( k=0; k<m ;k+=b) {
int i;
for(i=0; i<m; i+=b) {
int j;
for (j=0; j<m; j+=b) {
TASK_GEMM(OMPSSBLAS_NTRANSP, OMPSSBLAS_TRANSP, b, b, b, FP_MONE, &A[k*m+i], m, &B[k*m+j], m, FP_ONE, &C[j*m+i], m, 1);
}
}
}
#pragma omp taskwait
}
