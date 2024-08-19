#include "test_aux.h"
#include "bblas_gemm.h"
#include "fpmatr.h"
#include "ompss_mm.h"
#include "mkl.h"
void gemm_test(FP* C, FP* D, int ldc, int n)
{
FP res = mat_relerr(D, C, ldc, n);
if ( res < THRES )
fprintf(stdout, "PASS\n");
else
fprintf(stdout, "Suspicious!!!\n");
fprintf(stdout, "norm(D-C)/norm(C): %e\n", res);
}
int main(int argc, char** argv)
{
int M = 600;
int N = 500;
int K = 550;
int LDA = 650;
int LDB = 625;
int LDC = 700;
int BB = 117;
int DD = 103;
int CC = 89;
FP alpha = 13.4;
FP beta  = 99.1;
FP *A = malloc(LDA * MAX(M,K) * sizeof(FP));
FP *B = malloc(LDB * MAX(N,K) * sizeof(FP));
FP *C = malloc(LDC * MAX(M,N) * sizeof(FP));
FP *D = malloc(LDC * MAX(M,N) * sizeof(FP));
srand48(time(NULL));
int i;
for ( i = 0; i < LDA * MAX(M,K); ++i ) {
A[i] = (FP)drand48();
}
for ( i = 0; i < LDB * MAX(N,K); ++i ) {
B[i] = (FP)drand48();
}
for ( i = 0; i < LDC * MAX(M,N); ++i ) {
D[i] = C[i] = (FP)drand48();
}
for ( i = 0; i < 4; ++i ) {
int mkl0, mkl1;
ompssblas_t ompss0, ompss1;
switch (i){
case 0:
mkl0 = CblasNoTrans;
mkl1 = CblasNoTrans;
ompss0 = OMPSSBLAS_NTRANSP;
ompss1 = OMPSSBLAS_NTRANSP;
CBLAS_GEMM(CblasColMajor, mkl0, mkl1, M, N, K, alpha, A, LDA, B, LDB, beta, C, LDC);
OMPSS_GEMM(ompss0, ompss1, BB, DD, CC, M, N, K, alpha, A, LDA, B, LDB, beta, D, LDC);
#pragma omp taskwait
printf("C = alpha * A * B + beta * C\n");
break;
case 1:
mkl0 = CblasTrans;
mkl1 = CblasNoTrans;
ompss0 = OMPSSBLAS_TRANSP;
ompss1 = OMPSSBLAS_NTRANSP;
CBLAS_GEMM(CblasColMajor, mkl0, mkl1, M, N, K, alpha, A, LDA, B, LDB, beta, C, LDC);
OMPSS_GEMM(ompss0, ompss1, BB, DD, CC, M, N, K, alpha, A, LDA, B, LDB, beta, D, LDC);
#pragma omp taskwait
printf("C = alpha * A^T * B + beta * C\n");
break;
case 2:
mkl0 = CblasNoTrans;
mkl1 = CblasTrans;
ompss0 = OMPSSBLAS_NTRANSP;
ompss1 = OMPSSBLAS_TRANSP;
CBLAS_GEMM(CblasColMajor, mkl0, mkl1, M, N, K, alpha, A, LDA, B, LDB, beta, C, LDC);
OMPSS_GEMM(ompss0, ompss1, BB, DD, CC, M, N, K, alpha, A, LDA, B, LDB, beta, D, LDC);
#pragma omp taskwait
printf("C = alpha * A * B^T + beta * C\n");
break;
case 3:
mkl0 = CblasTrans;
mkl1 = CblasTrans;
ompss0 = OMPSSBLAS_TRANSP;
ompss1 = OMPSSBLAS_TRANSP;
CBLAS_GEMM(CblasColMajor, mkl0, mkl1, M, N, K, alpha, A, LDA, B, LDB, beta, C, LDC);
OMPSS_GEMM(ompss0, ompss1, BB, DD, CC, M, N, K, alpha, A, LDA, B, LDB, beta, D, LDC);
#pragma omp taskwait
printf("C = alpha * A^T * B^T + beta * C\n");
break;
}
gemm_test(C, D, LDC, MAX(M,N));
}
free(A);
free(B);
free(C);
free(D);
return 0;
}
