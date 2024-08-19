#include "test_aux.h"
#include "fpmatr.h"
#include "ompss_lu.h"
#include "mkl.h"
void lu_check(FP* C, FP* D, int ldc, int n)
{
FP res = mat_relerr(D, C, ldc, n);
if ( res < THRES)
fprintf(stdout, "PASS\n");
else
fprintf(stdout, "Suspicious!!!\n");
fprintf(stdout, "norm(D-C)/norm(C): %e\n", res);
}
int main(int argc, char** argv)
{
int M = 530;
int N = 530;
int B = 33;
FP *C = malloc(M * N * sizeof(FP));
FP *D = malloc(M * N * sizeof(FP));
FP *Dl = malloc(M * N * sizeof(FP));
int *ipivC = calloc (MAX(M,N), sizeof(int));
int *ipivD = calloc (MAX(M,N), sizeof(int));
srand48(time(NULL));
int j;
for ( j = 0; j < N; ++j ) {
int i;
for ( i = 0; i < M; ++i ) {
Dl[j*M+i]=D[j*M+i]=C[j*M+i] = drand48();
if ( j == i ) {
C[j*M+i] += 100;
D[j*M+i] += 100;
Dl[j*M+i] += 100;
}
}
}
LAPACK_GETRF(LAPACK_COL_MAJOR, M, N, C, M, ipivC);
OMPSS_LU(M, N, B, D, M, ipivD);
#pragma omp taskwait
lu_check(D, C, M, N);
OMPSS_LULL(M, N, B, Dl, M, ipivD);
#pragma omp taskwait
lu_check(Dl, C, M, N);
free(C);
free(D);
return 0;
}
