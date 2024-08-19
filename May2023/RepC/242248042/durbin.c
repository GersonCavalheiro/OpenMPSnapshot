#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <polybench.h>
#include "durbin.h"
static
void init_array (int n,
DATA_TYPE POLYBENCH_2D(y,N,N,n,n),
DATA_TYPE POLYBENCH_2D(sum,N,N,n,n),
DATA_TYPE POLYBENCH_1D(alpha,N,n),
DATA_TYPE POLYBENCH_1D(beta,N,n),
DATA_TYPE POLYBENCH_1D(r,N,n))
{
int i, j;
for (i = 0; i < n; i++)
{
alpha[i] = i;
beta[i] = (i+1)/n/2.0;
r[i] = (i+1)/n/4.0;
for (j = 0; j < n; j++) {
y[i][j] = ((DATA_TYPE) i*j) / n;
sum[i][j] = ((DATA_TYPE) i*j) / n;
}
}
}
static
void print_array(int n,
DATA_TYPE POLYBENCH_1D(out,N,n))
{
int i;
for (i = 0; i < n; i++) {
fprintf (stderr, DATA_PRINTF_MODIFIER, out[i]);
if (i % 20 == 0) fprintf (stderr, "\n");
}
}
static
void kernel_durbin(int n,
DATA_TYPE POLYBENCH_2D(y,N,N,n,n),
DATA_TYPE POLYBENCH_2D(sum,N,N,n,n),
DATA_TYPE POLYBENCH_1D(alpha,N,n),
DATA_TYPE POLYBENCH_1D(beta,N,n),
DATA_TYPE POLYBENCH_1D(r,N,n),
DATA_TYPE POLYBENCH_1D(out,N,n))
{
int i, k;
#pragma scop
y[0][0] = r[0];
beta[0] = 1;
alpha[0] = r[0];
for (k = 1; k < _PB_N; k++)
{
beta[k] = beta[k-1] - alpha[k-1] * alpha[k-1] * beta[k-1];
sum[0][k] = r[k];
for (i = 0; i <= k - 1; i++)
sum[i+1][k] = sum[i][k] + r[k-i-1] * y[i][k-1];
alpha[k] = -sum[k][k] * beta[k];
for (i = 0; i <= k-1; i++)
y[i][k] = y[i][k-1] + alpha[k] * y[k-i-1][k-1];
y[k][k] = alpha[k];
}
for (i = 0; i < _PB_N; i++)
out[i] = y[i][_PB_N-1];
#pragma endscop
}
int main(int argc, char** argv)
{
int n = N;
POLYBENCH_2D_ARRAY_DECL(y, DATA_TYPE, N, N, n, n);
POLYBENCH_2D_ARRAY_DECL(sum, DATA_TYPE, N, N, n, n);
POLYBENCH_1D_ARRAY_DECL(alpha, DATA_TYPE, N, n);
POLYBENCH_1D_ARRAY_DECL(beta, DATA_TYPE, N, n);
POLYBENCH_1D_ARRAY_DECL(r, DATA_TYPE, N, n);
POLYBENCH_1D_ARRAY_DECL(out, DATA_TYPE, N, n);
init_array (n,
POLYBENCH_ARRAY(y),
POLYBENCH_ARRAY(sum),
POLYBENCH_ARRAY(alpha),
POLYBENCH_ARRAY(beta),
POLYBENCH_ARRAY(r));
polybench_start_instruments;
kernel_durbin (n,
POLYBENCH_ARRAY(y),
POLYBENCH_ARRAY(sum),
POLYBENCH_ARRAY(alpha),
POLYBENCH_ARRAY(beta),
POLYBENCH_ARRAY(r),
POLYBENCH_ARRAY(out));
polybench_stop_instruments;
polybench_print_instruments;
polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(out)));
POLYBENCH_FREE_ARRAY(y);
POLYBENCH_FREE_ARRAY(sum);
POLYBENCH_FREE_ARRAY(alpha);
POLYBENCH_FREE_ARRAY(beta);
POLYBENCH_FREE_ARRAY(r);
POLYBENCH_FREE_ARRAY(out);
return 0;
}
