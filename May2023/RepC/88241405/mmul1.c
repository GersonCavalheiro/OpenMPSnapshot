#include "const.h"
#include "stdio.h"
#include "string.h"
#include <omp.h>
void mmul1(float A[ni][nk], float B[nk][nj], float C[ni][nj])
{
int i, j, k;
memset(C, 0, sizeof(C[0][0]) * ni * nj);
#pragma omp parallel private(i, j, k) shared(A, B, C)
{
#pragma omp for schedule(static)
for (i=0; i<ni; i++) {
for (k=0; k<nk; k++) {
for (j=0; j<nj; j++) {
C[i][j] += A[i][k]*B[k][j];
}
}
}
}
}
