#include "matmul_setup.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "genmat.h"
#include "matfprint.h"
#include "fptype.h"
#define MAX(a,b) ((a)>(b)?(a):(b))
extern int lda;
extern int ldb;
extern int ldc;
int matmul_setup(const char *fname, int m, int n, int k, int b, int d, int c, void **A, void **B, void **C) 
{
*A = GENMAT(lda, MAX(m, k), lda);
double *ptrA = *A;
int sizeA = lda * MAX(m, k);
#pragma omp register ([sizeA]ptrA)
*B = GENMAT(ldb, MAX(k, n), ldb);
double *ptrB = *B;
int sizeB = ldb * MAX(k, n);
#pragma omp register ([sizeB]ptrB)
*C = malloc(ldc * MAX(m, n) * sizeof(fp_t));
return 0;
}
int matmul_shutdown(int m, int n, int k, int b, int d, int c, void *A, void *B, void *C) 
{
free(A);
free(B);
free(C);
return 0;
}
