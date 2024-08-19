#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <polybench.h>
#include "cholesky.h"
static
void init_array(int n,
DATA_TYPE POLYBENCH_1D(p,N,n),
DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
int i, j;
for (i = 0; i < n; i++)
{
p[i] = 1.0 / n;
for (j = 0; j < n; j++)
A[i][j] = 1.0 / n;
}
}
static
void print_array(int n,
DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
int i, j;
for (i = 0; i < n; i++)
for (j = 0; j < n; j++) {
fprintf (stderr, DATA_PRINTF_MODIFIER, A[i][j]);
if ((i * N + j) % 20 == 0) fprintf (stderr, "\n");
}
}
static
void kernel_cholesky(int n,
DATA_TYPE POLYBENCH_1D(p,N,n),
DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
int i, j, k;
DATA_TYPE x;
#pragma scop
for (i = 0; i < _PB_N; ++i)
{
x = A[i][i];
for (j = 0; j <= i - 1; ++j)
x = x - A[i][j] * A[i][j];
p[i] = 1.0 / sqrt(x);
for (j = i + 1; j < _PB_N; ++j)
{
x = A[i][j];
for (k = 0; k <= i - 1; ++k)
x = x - A[j][k] * A[i][k];
A[j][i] = x * p[i];
}
}
#pragma endscop
}
int main(int argc, char** argv)
{
int n = N;
POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
POLYBENCH_1D_ARRAY_DECL(p, DATA_TYPE, N, n);
init_array (n, POLYBENCH_ARRAY(p), POLYBENCH_ARRAY(A));
polybench_start_instruments;
kernel_cholesky (n, POLYBENCH_ARRAY(p), POLYBENCH_ARRAY(A));
polybench_stop_instruments;
polybench_print_instruments;
polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));
POLYBENCH_FREE_ARRAY(A);
POLYBENCH_FREE_ARRAY(p);
return 0;
}
