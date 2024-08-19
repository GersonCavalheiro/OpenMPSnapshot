#include "genmat.h"
#include "densutil.h"
#include "bblas_gemm.h"
#include "fpblas.h"
#include "fptype.h"
#include "blas.h"
#include <stdio.h>
int main(int argc, char *argv[]) {
int m = atoi(argv[1]);
int n = atoi(argv[2]);
int k = atoi(argv[3]);
int bm = atoi(argv[4]);
int bn = atoi(argv[5]);
int bk = atoi(argv[6]);
float *A = sgenmat(m, k);
float *B = sgenmat(k, n);
float *C = malloc(m * n * sizeof(float));
float *Cex = malloc(m * n * sizeof(float));
bblas_sgemm(bm, bn, bk, m, n, k, 1.0, A, B, 0.0, C);
#pragma omp taskwait
BLAS_gemm(MAT_NOTRANSP, MAT_NOTRANSP, m, n, k, FP_ONE, A, m, B, k, FP_NOUGHT, Cex, m);
float err = dmat_srelerr(m, n, Cex, C);
printf("err %.16e\n", err);
return 0;
}
