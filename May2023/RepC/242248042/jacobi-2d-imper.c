#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <polybench.h>
#include "jacobi-2d-imper.h"
static
void init_array (int n,
DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
DATA_TYPE POLYBENCH_2D(B,N,N,n,n))
{
int i, j;
for (i = 0; i < n; i++)
for (j = 0; j < n; j++)
{
A[i][j] = ((DATA_TYPE) i*(j+2) + 2) / n;
B[i][j] = ((DATA_TYPE) i*(j+3) + 3) / n;
}
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
void kernel_jacobi_2d_imper(int tsteps,
int n,
DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
DATA_TYPE POLYBENCH_2D(B,N,N,n,n))
{
int t, i, j;
#pragma scop
for (t = 0; t < _PB_TSTEPS; t++)
{
for (i = 1; i < _PB_N - 1; i++)
for (j = 1; j < _PB_N - 1; j++)
B[i][j] = 0.2 * (A[i][j] + A[i][j-1] + A[i][1+j] + A[1+i][j] + A[i-1][j]);
for (i = 1; i < _PB_N-1; i++)
for (j = 1; j < _PB_N-1; j++)
A[i][j] = B[i][j];
}
#pragma endscop
}
int main(int argc, char** argv)
{
int n = N;
int tsteps = TSTEPS;
POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);
init_array (n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));
polybench_start_instruments;
kernel_jacobi_2d_imper (tsteps, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));
polybench_stop_instruments;
polybench_print_instruments;
polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));
POLYBENCH_FREE_ARRAY(A);
POLYBENCH_FREE_ARRAY(B);
return 0;
}
