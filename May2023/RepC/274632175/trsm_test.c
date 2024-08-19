#include "test_aux.h"
#include "fpmatr.h"
#include "ompss_trsm.h"
#include "mkl.h"
void trsm_test(FP* C, FP* D, int ldc, int n)
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
int BB = 121;
int M = 600;
int N = 500;
int LDA = 650;
int LDC = 625;
FP alpha = -2.11;
FP *A = malloc(LDA * M * sizeof(FP));
FP *C = malloc(LDC * N * sizeof(FP));
FP *D = malloc(LDC * N * sizeof(FP));
srand48(time(NULL));
int i;
for ( i = 0; i < LDA * M; ++i ) {
A[i] = (FP)drand48()+100;
}
for ( i = 0; i < LDC * N; ++i ) {
D[i] = C[i] = (FP)drand48()+100;
}
for ( i = 0; i < 4; ++i ) {
int mkl0, mkl1, mkl2, mkl3;
ompssblas_t ompss0, ompss1, ompss2, ompss3;
switch (i){
case 0:
mkl0 = CblasLeft;
mkl1 = CblasLower;
mkl2 = CblasNoTrans;
mkl3 = CblasNonUnit;
ompss0 = OMPSSBLAS_LEFT;
ompss1 = OMPSSBLAS_LOWERTRIANG;
ompss2 = OMPSSBLAS_NTRANSP;
ompss3 = OMPSSBLAS_NDIAGUNIT;
CBLAS_TRSM(CblasColMajor, mkl0, mkl1, mkl2, mkl3, M, N, alpha, A, LDA, C, LDC);
OMPSS_TRSM(ompss0, ompss1, ompss2, ompss3, M, N, BB, alpha, A, LDA, D, LDC);
#pragma omp taskwait
printf("left lower ntrans\n");
break;
case 1:
mkl0 = CblasLeft;
mkl1 = CblasLower;
mkl2 = CblasTrans;
mkl3 = CblasNonUnit;
ompss0 = OMPSSBLAS_LEFT;
ompss1 = OMPSSBLAS_LOWERTRIANG;
ompss2 = OMPSSBLAS_TRANSP;
ompss3 = OMPSSBLAS_NDIAGUNIT;
CBLAS_TRSM(CblasColMajor, mkl0, mkl1, mkl2, mkl3, M, N, alpha, A, LDA, C, LDC);
OMPSS_TRSM(ompss0, ompss1, ompss2, ompss3, M, N, BB, alpha, A, LDA, D, LDC);
#pragma omp taskwait
printf("left lower trans\n");
break;
case 2:
mkl0 = CblasLeft;
mkl1 = CblasUpper;
mkl2 = CblasNoTrans;
mkl3 = CblasNonUnit;
ompss0 = OMPSSBLAS_LEFT;
ompss1 = OMPSSBLAS_UPPERTRIANG;
ompss2 = OMPSSBLAS_NTRANSP;
ompss3 = OMPSSBLAS_NDIAGUNIT;
CBLAS_TRSM(CblasColMajor, mkl0, mkl1, mkl2, mkl3, M, N, alpha, A, LDA, C, LDC);
OMPSS_TRSM(ompss0, ompss1, ompss2, ompss3, M, N, BB, alpha, A, LDA, D, LDC);
#pragma omp taskwait
printf("left upper ntrans\n");
break;
case 3:
mkl0 = CblasLeft;
mkl1 = CblasUpper;
mkl2 = CblasTrans;
mkl3 = CblasNonUnit;
ompss0 = OMPSSBLAS_LEFT;
ompss1 = OMPSSBLAS_UPPERTRIANG;
ompss2 = OMPSSBLAS_TRANSP;
ompss3 = OMPSSBLAS_NDIAGUNIT;
CBLAS_TRSM(CblasColMajor, mkl0, mkl1, mkl2, mkl3, M, N, alpha, A, LDA, C, LDC);
OMPSS_TRSM(ompss0, ompss1, ompss2, ompss3, M, N, BB, alpha, A, LDA, D, LDC);
#pragma omp taskwait
printf("left upper trans\n");
break;
}
trsm_test(C, D, LDC, N);
}
free(A);
free(C);
free(D);
return 0;
}
