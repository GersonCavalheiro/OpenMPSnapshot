#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <polybench.h>
#include "3mm.h"
static
void init_array(int ni, int nj, int nk, int nl, int nm,
DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
DATA_TYPE POLYBENCH_2D(C,NJ,NM,nj,nm),
DATA_TYPE POLYBENCH_2D(D,NM,NL,nm,nl))
{
int i, j;
for (i = 0; i < ni; i++)
for (j = 0; j < nk; j++)
A[i][j] = ((DATA_TYPE) i*j) / ni;
for (i = 0; i < nk; i++)
for (j = 0; j < nj; j++)
B[i][j] = ((DATA_TYPE) i*(j+1)) / nj;
for (i = 0; i < nj; i++)
for (j = 0; j < nm; j++)
C[i][j] = ((DATA_TYPE) i*(j+3)) / nl;
for (i = 0; i < nm; i++)
for (j = 0; j < nl; j++)
D[i][j] = ((DATA_TYPE) i*(j+2)) / nk;
}
static
void print_array(int ni, int nl,
DATA_TYPE POLYBENCH_2D(G,NI,NL,ni,nl))
{
int i, j;
for (i = 0; i < ni; i++)
for (j = 0; j < nl; j++) {
fprintf (stderr, DATA_PRINTF_MODIFIER, G[i][j]);
if ((i * ni + j) % 20 == 0) fprintf (stderr, "\n");
}
fprintf (stderr, "\n");
}
static
void kernel_3mm(int ni, int nj, int nk, int nl, int nm,
DATA_TYPE POLYBENCH_2D(E,NI,NJ,ni,nj),
DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
DATA_TYPE POLYBENCH_2D(F,NJ,NL,nj,nl),
DATA_TYPE POLYBENCH_2D(C,NJ,NM,nj,nm),
DATA_TYPE POLYBENCH_2D(D,NM,NL,nm,nl),
DATA_TYPE POLYBENCH_2D(G,NI,NL,ni,nl))
{
int i, j, k;
#pragma scop
for (i = 0; i < _PB_NI; i++)
for (j = 0; j < _PB_NJ; j++)
{
E[i][j] = 0;
for (k = 0; k < _PB_NK; ++k)
E[i][j] += A[i][k] * B[k][j];
}
for (i = 0; i < _PB_NJ; i++)
for (j = 0; j < _PB_NL; j++)
{
F[i][j] = 0;
for (k = 0; k < _PB_NM; ++k)
F[i][j] += C[i][k] * D[k][j];
}
for (i = 0; i < _PB_NI; i++)
for (j = 0; j < _PB_NL; j++)
{
G[i][j] = 0;
for (k = 0; k < _PB_NJ; ++k)
G[i][j] += E[i][k] * F[k][j];
}
#pragma endscop
}
int main(int argc, char** argv)
{
int ni = NI;
int nj = NJ;
int nk = NK;
int nl = NL;
int nm = NM;
POLYBENCH_2D_ARRAY_DECL(E, DATA_TYPE, NI, NJ, ni, nj);
POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NI, NK, ni, nk);
POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, NK, NJ, nk, nj);
POLYBENCH_2D_ARRAY_DECL(F, DATA_TYPE, NJ, NL, nj, nl);
POLYBENCH_2D_ARRAY_DECL(C, DATA_TYPE, NJ, NM, nj, nm);
POLYBENCH_2D_ARRAY_DECL(D, DATA_TYPE, NM, NL, nm, nl);
POLYBENCH_2D_ARRAY_DECL(G, DATA_TYPE, NI, NL, ni, nl);
init_array (ni, nj, nk, nl, nm,
POLYBENCH_ARRAY(A),
POLYBENCH_ARRAY(B),
POLYBENCH_ARRAY(C),
POLYBENCH_ARRAY(D));
polybench_start_instruments;
kernel_3mm (ni, nj, nk, nl, nm,
POLYBENCH_ARRAY(E),
POLYBENCH_ARRAY(A),
POLYBENCH_ARRAY(B),
POLYBENCH_ARRAY(F),
POLYBENCH_ARRAY(C),
POLYBENCH_ARRAY(D),
POLYBENCH_ARRAY(G));
polybench_stop_instruments;
polybench_print_instruments;
polybench_prevent_dce(print_array(ni, nl,  POLYBENCH_ARRAY(G)));
POLYBENCH_FREE_ARRAY(E);
POLYBENCH_FREE_ARRAY(A);
POLYBENCH_FREE_ARRAY(B);
POLYBENCH_FREE_ARRAY(F);
POLYBENCH_FREE_ARRAY(C);
POLYBENCH_FREE_ARRAY(D);
POLYBENCH_FREE_ARRAY(G);
return 0;
}
