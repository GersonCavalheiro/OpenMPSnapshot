


















#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <assert.h>
#include "util.h"



#pragma omp task inout([ts][ts]A)
void omp_potrf(double * const A, int ts, int ld)
{
static int INFO;
static char L = 'L';
dpotrf_(&L, &ts, A, &ld, &INFO);
}

#pragma omp task in([ts][ts]A) inout([ts][ts]B)
void omp_trsm(double *A, double *B, int ts, int ld)
{
static char LO = 'L', TR = 'T', NU = 'N', RI = 'R';
static double DONE = 1.0;
dtrsm_(&RI, &LO, &TR, &NU, &ts, &ts, &DONE, A, &ld, B, &ld );
}

#pragma omp task in([ts][ts]A) inout([ts][ts]B)
void omp_syrk(double *A, double *B, int ts, int ld)
{
static char LO = 'L', NT = 'N';
static double DONE = 1.0, DMONE = -1.0;
dsyrk_(&LO, &NT, &ts, &ts, &DMONE, A, &ld, &DONE, B, &ld );
}

void gemm_tile(double *A, double *B, double *C, int ts, int ld)
{
static char TR = 'T', NT = 'N';
static double DONE = 1.0, DMONE = -1.0;
dgemm_(&NT, &TR, &ts, &ts, &ts, &DMONE, A, &ld, B, &ld, &DONE, C, &ld);
}

#pragma omp task in([super][super]A, [super][super]B) inout([super][super]C)
void omp_gemm(double *A, double *B , double *C ,int super, int region)
{
int i, j, k;

for(k=0; k<super ;k+=region)
{
for(i=0; i<super;i+=region)
{
for(j=0; j<super;j+=region)
{
gemm_tile(&A[k*super+i], &B[k*super+j], &C[j*super+i], super, region);    
}
}
}
}

void cholesky_blocked(const int ts, const int nt, double* Ah[nt][nt])
{
for (int k = 0; k < nt; k++) {

omp_potrf (Ah[k][k], ts, ts);

for (int i = k + 1; i < nt; i++) {
omp_trsm (Ah[k][k], Ah[k][i], ts, ts);
}

for (int i = k + 1; i < nt; i++) {
for (int j = k + 1; j < i; j++) {
omp_gemm (Ah[k][i], Ah[k][j], Ah[j][i], ts, ts);
}
omp_syrk (Ah[k][i], Ah[i][i], ts, ts);
}
}
#pragma omp taskwait
}

int main(int argc, char* argv[])
{

const double eps = BLAS_dfpinfo( blas_eps );


if ( argc != 3) {
printf( "cholesky size block_size\n" );
exit( -1 );
}
const int n = atoi(argv[1]); 
const int ts = atoi(argv[2]);  

double * const matrix = (double *) malloc(n * n * sizeof(double));
assert(matrix != NULL);

initialize_matrix(n, ts, matrix);

double * const original_matrix = (double *) malloc(n * n * sizeof(double));
assert(original_matrix != NULL);

const int nt = n / ts;

double *Ah[nt][nt];

for (int i = 0; i < nt; i++) {
for (int j = 0; j < nt; j++) {
Ah[i][j] = malloc(ts * ts * sizeof(double));
assert(Ah[i][j] != NULL);
}
}

for (int i = 0; i < n * n; i++ ) {
original_matrix[i] = matrix[i];
}

printf ("Executing ...\n");
convert_to_blocks(ts, nt, n, (double(*)[n]) matrix, Ah);

const float t1 = get_time();
cholesky_blocked(ts, nt, (double* (*)[nt]) Ah);

const float t2 = get_time() - t1;
convert_to_linear(ts, nt, n, Ah, (double (*)[n]) matrix);

const char uplo = 'L';
const int info_factorization = check_factorization( n, original_matrix, matrix, n, uplo, eps);

free(original_matrix);

float time = t2;
float gflops = (((1.0 / 3.0) * n * n * n) / ((time) * 1.0e+9));

printf( "============ CHOLESKY RESULTS ============\n" );
printf( "  matrix size:          %dx%d\n", n, n);
printf( "  block size:           %dx%d\n", ts, ts);
printf( "  time (s):             %f\n", time);
printf( "  performance (gflops): %f\n", gflops);
printf( "==========================================\n" );

for (int i = 0; i < nt; i++) {
for (int j = 0; j < nt; j++) {
assert(Ah[i][j] != NULL);
free(Ah[i][j]);
}
}

free(matrix);
return 0;
}

