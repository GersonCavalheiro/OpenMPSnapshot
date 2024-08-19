#include "test_aux.h"
#include "fpmatr.h"
#include "ompss_dense_chol.h"
#include "mkl.h"
void potrf_check(FP* A, FP* A0, int N)
{
int i,j;
double eps;
eps = LAPACKE_dlamch_work('e');
double *L1 = calloc(N*N,sizeof(double));
double *L2 = calloc(N*N,sizeof(double));
double *Residual = calloc(N*N, sizeof(double));
double *work = malloc(N*N*sizeof(double));
LAPACK_LACPY(LAPACK_COL_MAJOR,' ', N, N, A0, N, Residual, N);
LAPACK_LACPY(LAPACK_COL_MAJOR,'l', N, N, A, N, L1, N);
LAPACK_LACPY(LAPACK_COL_MAJOR,'l', N, N, A, N, L2, N);
CBLAS_TRMM(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, N, N, 1.0, L1, N, L2, N);
for (i = 0; i < N; i++)
for (j = 0; j < N; j++)
Residual[j*N+i] = L2[j*N+i] - Residual[j*N+i];
FP Rnorm = LAPACK_LANGE(LAPACK_COL_MAJOR, 'I', N, N, Residual, N, work);
FP Anorm = LAPACK_LANGE(LAPACK_COL_MAJOR, 'I', N, N, A0, N, work);
fprintf(stderr, "-- ||L'L-A||_oo/(||A||_oo.N.eps) = %e\n", Rnorm/(Anorm*N*eps));
if ( isnan(Rnorm/(Anorm*N*eps)) || (Rnorm/(Anorm*N*eps) > 10.0) ){
printf("Suspicious\n");
} else {
printf("Pass\n");
}
}
int main(int argc, char** argv)
{
int N = 1989;
int B = 533;
FP *A = malloc(N * N * sizeof(FP));
FP *A0 = malloc(N * N * sizeof(FP));
GENMAT_SYM_FULL(N, N, A0);
int i;
for (i=0; i<N*N; ++i)
A[i]=A0[i];
OMPSS_CHOL(N, B, B, A, N);
#pragma omp taskwait
potrf_check(A, A0, N);
free(A); free(A0);
return 0;
}
