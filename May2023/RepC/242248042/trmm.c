#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <polybench.h>
#include "trmm.h"
static
void init_array(int ni,
DATA_TYPE *alpha,
DATA_TYPE POLYBENCH_2D(A,NI,NI,ni,ni),
DATA_TYPE POLYBENCH_2D(B,NI,NI,ni,ni))
{
int i, j;
*alpha = 32412;
for (i = 0; i < ni; i++)
for (j = 0; j < ni; j++) {
A[i][j] = ((DATA_TYPE) i*j) / ni;
B[i][j] = ((DATA_TYPE) i*j) / ni;
}
}
static
void print_array(int ni,
DATA_TYPE POLYBENCH_2D(B,NI,NI,ni,ni))
{
int i, j;
for (i = 0; i < ni; i++)
for (j = 0; j < ni; j++) {
fprintf (stderr, DATA_PRINTF_MODIFIER, B[i][j]);
if ((i * ni + j) % 20 == 0) fprintf (stderr, "\n");
}
fprintf (stderr, "\n");
}
static
void kernel_trmm(int ni,
DATA_TYPE alpha,
DATA_TYPE POLYBENCH_2D(A,NI,NI,ni,ni),
DATA_TYPE POLYBENCH_2D(B,NI,NI,ni,ni))
{
int i, j, k;
#pragma scop
for (i = 1; i < _PB_NI; i++)
for (j = 0; j < _PB_NI; j++)
for (k = 0; k < i; k++)
B[i][j] += alpha * A[i][k] * B[j][k];
#pragma endscop
}
int main(int argc, char** argv)
{
int ni = NI;
DATA_TYPE alpha;
POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NI,ni,ni);
POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,NI,NI,ni,ni);
init_array (ni, &alpha, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));
polybench_start_instruments;
kernel_trmm (ni, alpha, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));
polybench_stop_instruments;
polybench_print_instruments;
polybench_prevent_dce(print_array(ni, POLYBENCH_ARRAY(B)));
POLYBENCH_FREE_ARRAY(A);
POLYBENCH_FREE_ARRAY(B);
return 0;
}
