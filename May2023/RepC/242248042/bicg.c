#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <polybench.h>
#include "bicg.h"
static
void init_array (int nx, int ny,
DATA_TYPE POLYBENCH_2D(A,NX,NY,nx,ny),
DATA_TYPE POLYBENCH_1D(r,NX,nx),
DATA_TYPE POLYBENCH_1D(p,NY,ny))
{
int i, j;
for (i = 0; i < ny; i++)
p[i] = i * M_PI;
for (i = 0; i < nx; i++) {
r[i] = i * M_PI;
for (j = 0; j < ny; j++)
A[i][j] = ((DATA_TYPE) i*(j+1))/nx;
}
}
static
void print_array(int nx, int ny,
DATA_TYPE POLYBENCH_1D(s,NY,ny),
DATA_TYPE POLYBENCH_1D(q,NX,nx))
{
int i;
for (i = 0; i < ny; i++) {
fprintf (stderr, DATA_PRINTF_MODIFIER, s[i]);
if (i % 20 == 0) fprintf (stderr, "\n");
}
for (i = 0; i < nx; i++) {
fprintf (stderr, DATA_PRINTF_MODIFIER, q[i]);
if (i % 20 == 0) fprintf (stderr, "\n");
}
fprintf (stderr, "\n");
}
static
void kernel_bicg(int nx, int ny,
DATA_TYPE POLYBENCH_2D(A,NX,NY,nx,ny),
DATA_TYPE POLYBENCH_1D(s,NY,ny),
DATA_TYPE POLYBENCH_1D(q,NX,nx),
DATA_TYPE POLYBENCH_1D(p,NY,ny),
DATA_TYPE POLYBENCH_1D(r,NX,nx))
{
int i, j;
#pragma scop
for (i = 0; i < _PB_NY; i++)
s[i] = 0;
for (i = 0; i < _PB_NX; i++)
{
q[i] = 0;
for (j = 0; j < _PB_NY; j++)
{
s[j] = s[j] + r[i] * A[i][j];
q[i] = q[i] + A[i][j] * p[j];
}
}
#pragma endscop
}
int main(int argc, char** argv)
{
int nx = NX;
int ny = NY;
POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NX, NY, nx, ny);
POLYBENCH_1D_ARRAY_DECL(s, DATA_TYPE, NY, ny);
POLYBENCH_1D_ARRAY_DECL(q, DATA_TYPE, NX, nx);
POLYBENCH_1D_ARRAY_DECL(p, DATA_TYPE, NY, ny);
POLYBENCH_1D_ARRAY_DECL(r, DATA_TYPE, NX, nx);
init_array (nx, ny,
POLYBENCH_ARRAY(A),
POLYBENCH_ARRAY(r),
POLYBENCH_ARRAY(p));
polybench_start_instruments;
kernel_bicg (nx, ny,
POLYBENCH_ARRAY(A),
POLYBENCH_ARRAY(s),
POLYBENCH_ARRAY(q),
POLYBENCH_ARRAY(p),
POLYBENCH_ARRAY(r));
polybench_stop_instruments;
polybench_print_instruments;
polybench_prevent_dce(print_array(nx, ny, POLYBENCH_ARRAY(s), POLYBENCH_ARRAY(q)));
POLYBENCH_FREE_ARRAY(A);
POLYBENCH_FREE_ARRAY(s);
POLYBENCH_FREE_ARRAY(q);
POLYBENCH_FREE_ARRAY(p);
POLYBENCH_FREE_ARRAY(r);
return 0;
}
