#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <polybench.h>
#include "syrk.h"
static
void init_array(int ni, int nj,
DATA_TYPE *alpha,
DATA_TYPE *beta,
DATA_TYPE POLYBENCH_2D(C,NI,NI,ni,ni),
DATA_TYPE POLYBENCH_2D(A,NI,NJ,ni,nj))
{
int i, j;
*alpha = 32412;
*beta = 2123;
for (i = 0; i < ni; i++)
for (j = 0; j < nj; j++)
A[i][j] = ((DATA_TYPE) i*j) / ni;
for (i = 0; i < ni; i++)
for (j = 0; j < ni; j++)
C[i][j] = ((DATA_TYPE) i*j) / ni;
}
static
void print_array(int ni,
DATA_TYPE POLYBENCH_2D(C,NI,NI,ni,ni))
{
int i, j;
for (i = 0; i < ni; i++)
for (j = 0; j < ni; j++) {
fprintf (stderr, DATA_PRINTF_MODIFIER, C[i][j]);
if ((i * ni + j) % 20 == 0) fprintf (stderr, "\n");
}
fprintf (stderr, "\n");
}
static
void kernel_syrk(int ni, int nj,
DATA_TYPE alpha,
DATA_TYPE beta,
DATA_TYPE POLYBENCH_2D(C,NI,NI,ni,ni),
DATA_TYPE POLYBENCH_2D(A,NI,NJ,ni,nj))
{
int i, j, k;
#pragma scop
for (i = 0; i < _PB_NI; i++)
for (j = 0; j < _PB_NI; j++)
C[i][j] *= beta;
for (i = 0; i < _PB_NI; i++)
for (j = 0; j < _PB_NI; j++)
for (k = 0; k < _PB_NJ; k++)
C[i][j] += alpha * A[i][k] * A[j][k];
#pragma endscop
}
int main(int argc, char** argv)
{
int ni = NI;
int nj = NJ;
DATA_TYPE alpha;
DATA_TYPE beta;
POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,NI,NI,ni,ni);
POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NJ,ni,nj);
init_array (ni, nj, &alpha, &beta, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(A));
polybench_start_instruments;
kernel_syrk (ni, nj, alpha, beta, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(A));
polybench_stop_instruments;
polybench_print_instruments;
polybench_prevent_dce(print_array(ni, POLYBENCH_ARRAY(C)));
POLYBENCH_FREE_ARRAY(C);
POLYBENCH_FREE_ARRAY(A);
return 0;
}
