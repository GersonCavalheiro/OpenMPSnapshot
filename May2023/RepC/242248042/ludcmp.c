#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <polybench.h>
#include "ludcmp.h"
static
void init_array (int n,
DATA_TYPE POLYBENCH_2D(A,N+1,N+1,n+1,n+1),
DATA_TYPE POLYBENCH_1D(b,N+1,n+1),
DATA_TYPE POLYBENCH_1D(x,N+1,n+1),
DATA_TYPE POLYBENCH_1D(y,N+1,n+1))
{
int i, j;
for (i = 0; i <= n; i++)
{
x[i] = i + 1;
y[i] = (i+1)/n/2.0 + 1;
b[i] = (i+1)/n/2.0 + 42;
for (j = 0; j <= n; j++) {
A[i][j] = ((DATA_TYPE) (i+1)*(j+1)) / n;
}
}
}
static
void print_array(int n,
DATA_TYPE POLYBENCH_1D(x,N+1,n+1))
{
int i;
for (i = 0; i <= n; i++) {
fprintf (stderr, DATA_PRINTF_MODIFIER, x[i]);
if (i % 20 == 0) fprintf (stderr, "\n");
}
}
static
void kernel_ludcmp(int n,
DATA_TYPE POLYBENCH_2D(A,N+1,N+1,n+1,n+1),
DATA_TYPE POLYBENCH_1D(b,N+1,n+1),
DATA_TYPE POLYBENCH_1D(x,N+1,n+1),
DATA_TYPE POLYBENCH_1D(y,N+1,n+1))
{
int i, j, k;
DATA_TYPE w;
#pragma scop
b[0] = 1.0;
for (i = 0; i < _PB_N; i++)
{
for (j = i+1; j <= _PB_N; j++)
{
w = A[j][i];
for (k = 0; k < i; k++)
w = w- A[j][k] * A[k][i];
A[j][i] = w / A[i][i];
}
for (j = i+1; j <= _PB_N; j++)
{
w = A[i+1][j];
for (k = 0; k <= i; k++)
w = w  - A[i+1][k] * A[k][j];
A[i+1][j] = w;
}
}
y[0] = b[0];
for (i = 1; i <= _PB_N; i++)
{
w = b[i];
for (j = 0; j < i; j++)
w = w - A[i][j] * y[j];
y[i] = w;
}
x[_PB_N] = y[_PB_N] / A[_PB_N][_PB_N];
for (i = 0; i <= _PB_N - 1; i++)
{
w = y[_PB_N - 1 - (i)];
for (j = _PB_N - i; j <= _PB_N; j++)
w = w - A[_PB_N - 1 - i][j] * x[j];
x[_PB_N - 1 - i] = w / A[_PB_N - 1 - (i)][_PB_N - 1-(i)];
}
#pragma endscop
}
int main(int argc, char** argv)
{
int n = N;
POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N+1, N+1, n+1, n+1);
POLYBENCH_1D_ARRAY_DECL(b, DATA_TYPE, N+1, n+1);
POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N+1, n+1);
POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N+1, n+1);
init_array (n,
POLYBENCH_ARRAY(A),
POLYBENCH_ARRAY(b),
POLYBENCH_ARRAY(x),
POLYBENCH_ARRAY(y));
polybench_start_instruments;
kernel_ludcmp (n,
POLYBENCH_ARRAY(A),
POLYBENCH_ARRAY(b),
POLYBENCH_ARRAY(x),
POLYBENCH_ARRAY(y));
polybench_stop_instruments;
polybench_print_instruments;
polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(x)));
POLYBENCH_FREE_ARRAY(A);
POLYBENCH_FREE_ARRAY(b);
POLYBENCH_FREE_ARRAY(x);
POLYBENCH_FREE_ARRAY(y);
return 0;
}
