#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <polybench.h>
#include "seidel-2d.h"
static
void init_array (int n,
DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
int i, j;
for (i = 0; i < n; i++)
for (j = 0; j < n; j++)
A[i][j] = ((DATA_TYPE) i*(j+2) + 2) / n;
}
static
void print_array(int n,
DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
int i, j;
for (i = 0; i < n; i++)
for (j = 0; j < n; j++) {
fprintf(stderr, DATA_PRINTF_MODIFIER, A[i][j]);
if ((i * n + j) % 20 == 0) fprintf(stderr, "\n");
}
fprintf(stderr, "\n");
}
static
void kernel_seidel_2d(int tsteps,
int n,
DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
int t, i, j;
#pragma scop
for (t = 0; t <= _PB_TSTEPS - 1; t++)
for (i = 1; i<= _PB_N - 2; i++)
for (j = 1; j <= _PB_N - 2; j++)
A[i][j] = (A[i-1][j-1] + A[i-1][j] + A[i-1][j+1]
+ A[i][j-1] + A[i][j] + A[i][j+1]
+ A[i+1][j-1] + A[i+1][j] + A[i+1][j+1])/9.0;
#pragma endscop
}
int main(int argc, char** argv)
{
int n = N;
int tsteps = TSTEPS;
POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
init_array (n, POLYBENCH_ARRAY(A));
polybench_start_instruments;
kernel_seidel_2d (tsteps, n, POLYBENCH_ARRAY(A));
polybench_stop_instruments;
polybench_print_instruments;
polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));
POLYBENCH_FREE_ARRAY(A);
return 0;
}
