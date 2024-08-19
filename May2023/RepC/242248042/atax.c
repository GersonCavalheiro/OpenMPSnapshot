#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <polybench.h>
#include "atax.h"
static
void init_array (int nx, int ny,
DATA_TYPE POLYBENCH_2D(A,NX,NY,nx,ny),
DATA_TYPE POLYBENCH_1D(x,NY,ny))
{
int i, j;
for (i = 0; i < ny; i++)
x[i] = i * M_PI;
for (i = 0; i < nx; i++)
for (j = 0; j < ny; j++)
A[i][j] = ((DATA_TYPE) i*(j+1)) / nx;
}
static
void print_array(int nx,
DATA_TYPE POLYBENCH_1D(y,NX,nx))
{
int i;
for (i = 0; i < nx; i++) {
fprintf (stderr, DATA_PRINTF_MODIFIER, y[i]);
if (i % 20 == 0) fprintf (stderr, "\n");
}
fprintf (stderr, "\n");
}
static
void kernel_atax(int nx, int ny,
DATA_TYPE POLYBENCH_2D(A,NX,NY,nx,ny),
DATA_TYPE POLYBENCH_1D(x,NY,ny),
DATA_TYPE POLYBENCH_1D(y,NY,ny),
DATA_TYPE POLYBENCH_1D(tmp,NX,nx))
{
int i, j;
#pragma scop
for (i = 0; i < _PB_NY; i++)
y[i] = 0;
for (i = 0; i < _PB_NX; i++)
{
tmp[i] = 0;
for (j = 0; j < _PB_NY; j++)
tmp[i] = tmp[i] + A[i][j] * x[j];
for (j = 0; j < _PB_NY; j++)
y[j] = y[j] + A[i][j] * tmp[i];
}
#pragma endscop
}
int main(int argc, char** argv)
{
int nx = NX;
int ny = NY;
POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NX, NY, nx, ny);
POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, NY, ny);
POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, NY, ny);
POLYBENCH_1D_ARRAY_DECL(tmp, DATA_TYPE, NX, nx);
init_array (nx, ny, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(x));
polybench_start_instruments;
kernel_atax (nx, ny,
POLYBENCH_ARRAY(A),
POLYBENCH_ARRAY(x),
POLYBENCH_ARRAY(y),
POLYBENCH_ARRAY(tmp));
polybench_stop_instruments;
polybench_print_instruments;
polybench_prevent_dce(print_array(nx, POLYBENCH_ARRAY(y)));
POLYBENCH_FREE_ARRAY(A);
POLYBENCH_FREE_ARRAY(x);
POLYBENCH_FREE_ARRAY(y);
POLYBENCH_FREE_ARRAY(tmp);
return 0;
}
