#include "spdsolv_setup.h"
#include <stdlib.h>
#include <time.h>
#include "fptype.h"
#include "fpblas.h"
#include "genmat.h"
#include "matfprint.h"
int spdsolv_setup(int check, int m, int n, int b, fp_t **A, fp_t **B, fp_t **X)
{ 
int m2 = m * m;
fp_t *lA;
lA = GENMAT_SPD(m, m);
#pragma omp register ([m*m]lA)
*A = lA;
int mn = m * n;
fp_t *lX = *X = GENMAT(m, n, m);
fp_t *lB = *B = malloc(mn * sizeof(fp_t));
#pragma omp register ([mn]lB)
BLAS_gemm(OMPSSBLAS_NTRANSP, OMPSSBLAS_NTRANSP, m, n, m, FP_ONE, lA, m, lX, m, FP_NOUGHT, lB, m);
if ( ! check ) {
free(lX);
*X = NULL;
}
return 0;
}
void spdsolv_shutdown(fp_t *A, fp_t *B, fp_t *X) 
{
free(A);
free(B);
free(X);
}
