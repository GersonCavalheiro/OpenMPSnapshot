#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <polybench.h>
#include "trisolv.h"
static
void init_array(int n,
DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
DATA_TYPE POLYBENCH_1D(x,N,n),
DATA_TYPE POLYBENCH_1D(c,N,n))
{
int i, j;
for (i = 0; i < n; i++)
{
c[i] = x[i] = ((DATA_TYPE) i) / n;
for (j = 0; j < n; j++)
A[i][j] = ((DATA_TYPE) i*j) / n;
}
}
static
void print_array(int n,
DATA_TYPE POLYBENCH_1D(x,N,n))
{
int i;
for (i = 0; i < n; i++) {
fprintf (stderr, DATA_PRINTF_MODIFIER, x[i]);
if (i % 20 == 0) fprintf (stderr, "\n");
}
}
static
void kernel_trisolv(int n,
DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
DATA_TYPE POLYBENCH_1D(x,N,n),
DATA_TYPE POLYBENCH_1D(c,N,n))
{
int i, j;
#pragma scop
for (i = 0; i < _PB_N; i++)
{
x[i] = c[i];
for (j = 0; j <= i - 1; j++)
x[i] = x[i] - A[i][j] * x[j];
x[i] = x[i] / A[i][i];
}
#pragma endscop
}
int main(int argc, char** argv)
{
int n = N;
POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
POLYBENCH_1D_ARRAY_DECL(c, DATA_TYPE, N, n);
init_array (n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(c));
polybench_start_instruments;
kernel_trisolv (n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(c));
polybench_stop_instruments;
polybench_print_instruments;
polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(x)));
POLYBENCH_FREE_ARRAY(A);
POLYBENCH_FREE_ARRAY(x);
POLYBENCH_FREE_ARRAY(c);
return 0;
}
