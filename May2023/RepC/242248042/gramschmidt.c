#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <polybench.h>
#include "gramschmidt.h"
static
void init_array(int ni, int nj,
DATA_TYPE POLYBENCH_2D(A,NI,NJ,ni,nj),
DATA_TYPE POLYBENCH_2D(R,NJ,NJ,nj,nj),
DATA_TYPE POLYBENCH_2D(Q,NI,NJ,ni,nj))
{
int i, j;
for (i = 0; i < ni; i++)
for (j = 0; j < nj; j++) {
A[i][j] = ((DATA_TYPE) i*j) / ni;
Q[i][j] = ((DATA_TYPE) i*(j+1)) / nj;
}
for (i = 0; i < nj; i++)
for (j = 0; j < nj; j++)
R[i][j] = ((DATA_TYPE) i*(j+2)) / nj;
}
static
void print_array(int ni, int nj,
DATA_TYPE POLYBENCH_2D(A,NI,NJ,ni,nj),
DATA_TYPE POLYBENCH_2D(R,NJ,NJ,nj,nj),
DATA_TYPE POLYBENCH_2D(Q,NI,NJ,ni,nj))
{
int i, j;
for (i = 0; i < ni; i++)
for (j = 0; j < nj; j++) {
fprintf (stderr, DATA_PRINTF_MODIFIER, A[i][j]);
if (i % 20 == 0) fprintf (stderr, "\n");
}
fprintf (stderr, "\n");
for (i = 0; i < nj; i++)
for (j = 0; j < nj; j++) {
fprintf (stderr, DATA_PRINTF_MODIFIER, R[i][j]);
if (i % 20 == 0) fprintf (stderr, "\n");
}
fprintf (stderr, "\n");
for (i = 0; i < ni; i++)
for (j = 0; j < nj; j++) {
fprintf (stderr, DATA_PRINTF_MODIFIER, Q[i][j]);
if (i % 20 == 0) fprintf (stderr, "\n");
}
fprintf (stderr, "\n");
}
static
void kernel_gramschmidt(int ni, int nj,
DATA_TYPE POLYBENCH_2D(A,NI,NJ,ni,nj),
DATA_TYPE POLYBENCH_2D(R,NJ,NJ,nj,nj),
DATA_TYPE POLYBENCH_2D(Q,NI,NJ,ni,nj))
{
int i, j, k;
DATA_TYPE nrm;
#pragma scop
for (k = 0; k < _PB_NJ; k++)
{
nrm = 0;
for (i = 0; i < _PB_NI; i++)
nrm += A[i][k] * A[i][k];
R[k][k] = sqrt(nrm);
for (i = 0; i < _PB_NI; i++)
Q[i][k] = A[i][k] / R[k][k];
for (j = k + 1; j < _PB_NJ; j++)
{
R[k][j] = 0;
for (i = 0; i < _PB_NI; i++)
R[k][j] += Q[i][k] * A[i][j];
for (i = 0; i < _PB_NI; i++)
A[i][j] = A[i][j] - Q[i][k] * R[k][j];
}
}
#pragma endscop
}
int main(int argc, char** argv)
{
int ni = NI;
int nj = NJ;
POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NJ,ni,nj);
POLYBENCH_2D_ARRAY_DECL(R,DATA_TYPE,NJ,NJ,nj,nj);
POLYBENCH_2D_ARRAY_DECL(Q,DATA_TYPE,NI,NJ,ni,nj);
init_array (ni, nj,
POLYBENCH_ARRAY(A),
POLYBENCH_ARRAY(R),
POLYBENCH_ARRAY(Q));
polybench_start_instruments;
kernel_gramschmidt (ni, nj,
POLYBENCH_ARRAY(A),
POLYBENCH_ARRAY(R),
POLYBENCH_ARRAY(Q));
polybench_stop_instruments;
polybench_print_instruments;
polybench_prevent_dce(print_array(ni, nj, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(R), POLYBENCH_ARRAY(Q)));
POLYBENCH_FREE_ARRAY(A);
POLYBENCH_FREE_ARRAY(R);
POLYBENCH_FREE_ARRAY(Q);
return 0;
}
