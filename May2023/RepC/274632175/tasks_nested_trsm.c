#include "tasks_nested_trsm.h"
#include "task_gemm.h"
#include "tasks_syrk.h"
#include "tasks_trsm.h"
#include "tasks_potrf.h"
#include "fptype.h"
#include "fpblas.h"
#include "fpmatr.h"
#include "blas.h"
#include "matfprint.h"
#ifdef DOUBLE_PRECISION
#define __t_ntrsm			ntask_dtrsm
#else
#define __t_ntrsm			ntask_strsm
#endif
void __t_ntrsm( int m, int b, int t, fp_t *A, fp_t *B) {
int k;
for (k=0; k<m; k+=b) {
int i;
for (i=0; i<m ; i+=b) {
TASK_TRSM(OMPSSBLAS_RIGHT, OMPSSBLAS_LOWERTRIANG, OMPSSBLAS_TRANSP, OMPSSBLAS_NDIAGUNIT, b, b, FP_ONE, &A[k*m+k], m, &B[k*m+i], m, 1);
int j;
for (j=k+b; j<m; j+=b) {
TASK_GEMM(OMPSSBLAS_NTRANSP, OMPSSBLAS_TRANSP, b, b, b, FP_MONE, &B[k*m+i], m, &A[k*m+j], m, FP_ONE, &B[j*m+i], m, 1);
}
}
}
#pragma omp taskwait
}
