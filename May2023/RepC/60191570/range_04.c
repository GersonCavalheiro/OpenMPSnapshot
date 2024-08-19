#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <libgen.h>
float * allocate_clean_block(int submatrix_size);
void lu0(float *diag, int submatrix_size);
void bdiv(float *diag, float *row, int submatrix_size);
void bmod(float *row, float *col, float *inner, int submatrix_size);
void fwd(float *diag, float *col, int submatrix_size);
const int MS = 1024;
const int BS = 128;
void sparselu_par_call(float **BENCH)
{
int ii, jj, kk;
#pragma omp parallel private(kk,ii,jj) shared(BENCH)
#pragma omp single
{
for (kk=0; kk<MS; kk++)
{
#pragma omp task firstprivate(kk) shared(BENCH) depend(inout: BENCH[kk*MS+kk:BS*BS])
lu0(BENCH[kk*MS+kk], BS);
#pragma analysis_check assert range(kk:0:1023:0)
for (jj=kk+1; jj<MS; jj++)
{
#pragma analysis_check assert range(jj:1:1023:0)
if (BENCH[kk*MS+jj] != NULL)
{
#pragma omp task firstprivate(kk, jj) shared(BENCH) depend(in: BENCH[kk*MS+kk:BS*BS]) depend(inout: BENCH[kk*MS+jj:BS*BS])
fwd(BENCH[kk*MS+kk], BENCH[kk*MS+jj], BS);
}
}
for (ii=kk+1; ii<MS; ii++)
{
#pragma analysis_check assert range(ii:1:1023:0)
if (BENCH[ii*MS+kk] != NULL)
{
#pragma omp task firstprivate(kk, ii) shared(BENCH) depend(in: BENCH[kk*MS+kk:BS*BS]) depend(inout: BENCH[ii*MS+kk:BS*BS])
bdiv (BENCH[kk*MS+kk], BENCH[ii*MS+kk], BS);
}
}
for (ii=kk+1; ii<MS; ii++)
{
#pragma analysis_check assert range(ii:1:1023:0)
if (BENCH[ii*MS+kk] != NULL)
for (jj=kk+1; jj<MS; jj++)
{
#pragma analysis_check assert range(jj:1:1023:0)
if (BENCH[kk*MS+jj] != NULL)
{
if (BENCH[ii*MS+jj]==NULL) BENCH[ii*MS+jj] = allocate_clean_block(BS);
#pragma omp task firstprivate(kk, jj, ii) shared(BENCH) depend(in: BENCH[ii*MS+kk:BS*BS], BENCH[kk*MS+jj:BS*BS]) depend(inout: BENCH[ii*MS+jj:BS*BS])
bmod(BENCH[ii*MS+kk], BENCH[kk*MS+jj], BENCH[ii*MS+jj], BS);
}
}
}
}
}
}
