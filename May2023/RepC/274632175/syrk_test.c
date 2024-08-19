#include "test_aux.h"
#include "fpmatr.h"
#include "ompss_syrk.h"
#include "mkl.h"
void syrk_test(FP* C, FP* D, int ldc, int n)
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
int BB = 123;
int N = 600;
int K = 500;
int LDA = 650;
int LDC = 625;
FP alpha = 13.4;
FP beta  = 99.1;
FP *A = malloc(LDA * MAX(N,K) * sizeof(FP));
FP *C = malloc(LDC * N * sizeof(FP));
FP *D = malloc(LDC * N * sizeof(FP));
srand48(time(NULL));
int i;
for ( i = 0; i < LDA * MAX(N,K); ++i ) {
A[i] = (FP)drand48();
}
for ( i = 0; i < LDC * N; ++i ) {
D[i] = C[i] = (FP)drand48();
}
for ( i = 0; i < 4; ++i ) {
int mkl0, mkl1;
ompssblas_t ompss0, ompss1;
switch (i){
case 0:
mkl0 = CblasLower;
mkl1 = CblasNoTrans;
ompss0 = OMPSSBLAS_LOWERTRIANG;
ompss1 = OMPSSBLAS_NTRANSP;
CBLAS_SYRK(CblasColMajor, mkl0, mkl1, N, K, alpha, A, LDA, beta, C, LDC);
OMPSS_SYRK(ompss0, ompss1, BB, N, K, alpha, A, LDA, beta, D, LDC);
#pragma omp taskwait
printf("lower ntrans\n");
break;
case 1:
mkl0 = CblasUpper;
mkl1 = CblasNoTrans;
ompss0 = OMPSSBLAS_UPPERTRIANG;
ompss1 = OMPSSBLAS_NTRANSP;
CBLAS_SYRK(CblasColMajor, mkl0, mkl1, N, K, alpha, A, LDA, beta, C, LDC);
OMPSS_SYRK(ompss0, ompss1, BB, N, K, alpha, A, LDA, beta, D, LDC);
#pragma omp taskwait
printf("upper ntrans\n");
break;
case 2:
mkl0 = CblasLower;
mkl1 = CblasTrans;
ompss0 = OMPSSBLAS_LOWERTRIANG;
ompss1 = OMPSSBLAS_TRANSP;
CBLAS_SYRK(CblasColMajor, mkl0, mkl1, N, K, alpha, A, LDA, beta, C, LDC);
OMPSS_SYRK(ompss0, ompss1, BB, N, K, alpha, A, LDA, beta, D, LDC);
#pragma omp taskwait
printf("lower trans\n");
break;
case 3:
mkl0 = CblasUpper;
mkl1 = CblasTrans;
ompss0 = OMPSSBLAS_UPPERTRIANG;
ompss1 = OMPSSBLAS_TRANSP;
CBLAS_SYRK(CblasColMajor, mkl0, mkl1, N, K, alpha, A, LDA, beta, C, LDC);
OMPSS_SYRK(ompss0, ompss1, BB, N, K, alpha, A, LDA, beta, D, LDC);
#pragma omp taskwait
printf("upper trans\n");
break;
}
syrk_test(C, D, LDC, N);
}
free(A);
free(C);
free(D);
return 0;
}
