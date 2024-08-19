#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <polybench.h>
#include "lu.h"
static
void init_array (int n,
DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
int i, j;
for (i = 0; i < n; i++)
for (j = 0; j < n; j++)
A[i][j] = ((DATA_TYPE) (i+1)*(j+1)) / n;
}
static
void print_array(int n,
DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
int i, j;
for (i = 0; i < n; i++)
for (j = 0; j < n; j++) {
fprintf (stderr, DATA_PRINTF_MODIFIER, A[i][j]);
if ((i * n + j) % 20 == 0) fprintf (stderr, "\n");
}
fprintf (stderr, "\n");
}
static
void kernel_lu(int n,
DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
int i, j, k;
#pragma scop
for (k = 0; k < _PB_N; k++)
{
for (j = k + 1; j < _PB_N; j++)
A[k][j] = A[k][j] / A[k][k];
for(i = k + 1; i < _PB_N; i++)
for (j = k + 1; j < _PB_N; j++)
A[i][j] = A[i][j] - A[i][k] * A[k][j];
}
#pragma endscop
}
int main(int argc, char** argv)
{
int n = N;
POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
init_array (n, POLYBENCH_ARRAY(A));
polybench_start_instruments;
kernel_lu (n, POLYBENCH_ARRAY(A));
polybench_stop_instruments;
polybench_print_instruments;
polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));
POLYBENCH_FREE_ARRAY(A);
return 0;
}
