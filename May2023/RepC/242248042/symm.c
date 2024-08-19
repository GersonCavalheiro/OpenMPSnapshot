#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <polybench.h>
#include "symm.h"
static
void init_array(int ni, int nj,
DATA_TYPE *alpha,
DATA_TYPE *beta,
DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj),
DATA_TYPE POLYBENCH_2D(A,NJ,NJ,nj,nj),
DATA_TYPE POLYBENCH_2D(B,NI,NJ,ni,nj))
{
int i, j;
*alpha = 32412;
*beta = 2123;
for (i = 0; i < ni; i++)
for (j = 0; j < nj; j++) {
C[i][j] = ((DATA_TYPE) i*j) / ni;
B[i][j] = ((DATA_TYPE) i*j) / ni;
}
for (i = 0; i < nj; i++)
for (j = 0; j < nj; j++)
A[i][j] = ((DATA_TYPE) i*j) / ni;
}
static
void print_array(int ni, int nj,
DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj))
{
int i, j;
for (i = 0; i < ni; i++)
for (j = 0; j < nj; j++) {
fprintf (stderr, DATA_PRINTF_MODIFIER, C[i][j]);
if ((i * ni + j) % 20 == 0) fprintf (stderr, "\n");
}
fprintf (stderr, "\n");
}
static
void kernel_symm(int ni, int nj,
DATA_TYPE alpha,
DATA_TYPE beta,
DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj),
DATA_TYPE POLYBENCH_2D(A,NJ,NJ,nj,nj),
DATA_TYPE POLYBENCH_2D(B,NI,NJ,ni,nj))
{
int i, j, k;
DATA_TYPE acc;
#pragma scop
for (i = 0; i < _PB_NI; i++)
for (j = 0; j < _PB_NJ; j++)
{
acc = 0;
for (k = 0; k < j - 1; k++)
{
C[k][j] += alpha * A[k][i] * B[i][j];
acc += B[k][j] * A[k][i];
}
C[i][j] = beta * C[i][j] + alpha * A[i][i] * B[i][j] + alpha * acc;
}
#pragma endscop
}
int main(int argc, char** argv)
{
int ni = NI;
int nj = NJ;
DATA_TYPE alpha;
DATA_TYPE beta;
POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,NI,NJ,ni,nj);
POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NJ,NJ,nj,nj);
POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,NI,NJ,ni,nj);
init_array (ni, nj, &alpha, &beta,
POLYBENCH_ARRAY(C),
POLYBENCH_ARRAY(A),
POLYBENCH_ARRAY(B));
polybench_start_instruments;
kernel_symm (ni, nj,
alpha, beta,
POLYBENCH_ARRAY(C),
POLYBENCH_ARRAY(A),
POLYBENCH_ARRAY(B));
polybench_stop_instruments;
polybench_print_instruments;
polybench_prevent_dce(print_array(ni, nj,  POLYBENCH_ARRAY(C)));
POLYBENCH_FREE_ARRAY(C);
POLYBENCH_FREE_ARRAY(A);
POLYBENCH_FREE_ARRAY(B);
return 0;
}
