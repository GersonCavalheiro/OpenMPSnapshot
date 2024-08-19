#include "const.h"
#include "stdio.h"
#include "string.h"
#include <stdlib.h>
#include <omp.h>
#define BS 64
void mmul2(float A[ni][nk], float B[nk][nj], float C[ni][nj])
{
int i, j, k, ii, jj;
float buff[BS][BS];
#pragma omp parallel private (i, j, k, ii, jj, buff) shared(A, B, C)
{
#pragma omp for schedule(static)
for (i = 0; i < ni/BS; ++i)
for (j = 0; j < nj/BS; ++j)
{
for (ii = 0; ii < BS; ++ii)
for (jj=0; jj<BS; ++jj) 
buff[ii][jj]=0;
for (ii = 0; ii < BS; ++ii)
for(k=0; k<nk; ++k) 
for (jj = 0; jj < BS; ++jj) 
buff[ii][jj] += A[i*BS+ii][k] * B[k][j*BS+jj];
for (ii = 0; ii < BS; ++ii)
for (jj=0; jj<BS; ++jj)
C[i*BS+ii][j*BS+jj] = buff[ii][jj];
}
} 
}
