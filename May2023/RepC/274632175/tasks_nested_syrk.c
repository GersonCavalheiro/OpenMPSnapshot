#include "tasks_nested_syrk.h"
#include "task_gemm.h"
#include "tasks_syrk.h"
#include "tasks_trsm.h"
#include "tasks_potrf.h"
#include "fptype.h"
#include "fpblas.h"
#include "blas.h"
#include "matfprint.h"
#ifdef DOUBLE_PRECISION
#define __t_nsyrk			ntask_dsyrk
#else
#define __t_nsyrk			ntask_ssyrk
#endif
#define DRYRUN	0
void __t_nsyrk(int m, int b, int t, fp_t *A, fp_t *C) {
int k;
for (k=0; k<m; k+=b) {
int j;
for (j=0; j<m; j+=b) {
TASK_SYRK(OMPSSBLAS_LOWERTRIANG, OMPSSBLAS_NTRANSP, b, b, FP_MONE, &A[j*m+k], m, FP_ONE, &C[k*m+k], m, 1);
}
int i;
for (i=0; i<k; i+=b) {
int j;
for (j=0; j<m; j+=b) {
TASK_GEMM(OMPSSBLAS_NTRANSP, OMPSSBLAS_TRANSP, b, b, b, FP_MONE, &A[j*m+k], m, &A[j*m+i], m, FP_ONE, &C[i*m+k], m, 1);
}
}
}
#pragma omp taskwait
}
