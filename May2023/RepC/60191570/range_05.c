#include <limits.h>
void omp_potrf(double * const A, int ts, int ld);
void omp_trsm(double *A, double *B, int ts, int ld);
void omp_syrk(double *A, double *B, int ts, int ld);
void omp_gemm(double *A, double *B, double *C, int ts, int ld);
long int long_max = LONG_MAX;
void cholesky_1(const int ts, const int nt, double* Ah[nt][nt])
{
int i, j, k;
#pragma analysis_check assert range(k:0:long_max:0)
for (k = 0; k < nt; k++) {
#pragma omp task untied inout(Ah[k][k]) 
omp_potrf (Ah[k][k], ts, ts);
#pragma analysis_check assert range(i:1:long_max:0)
for (i = k + 1; i < nt; i++) {
#pragma omp task untied in(Ah[k][k]) inout(Ah[k][i]) 
omp_trsm (Ah[k][k], Ah[k][i], ts, ts);
}
#pragma analysis_check assert range(i:1:long_max:0)
for (i = k + 1; i < nt; i++) {
#pragma analysis_check assert range(j:1:long_max:0)
for (j = k + 1; j < i; j++) {
#pragma omp task untied in(Ah[k][i], Ah[k][j]) inout(Ah[j][i]) 
omp_gemm (Ah[k][i], Ah[k][j], Ah[j][i], ts, ts);
}
#pragma omp task untied in(Ah[k][i]) inout(Ah[i][i]) 
omp_syrk (Ah[k][i], Ah[i][i], ts, ts);
}
}
#pragma omp taskwait
}
const int NT = 256;
const int TS = 32;
void cholesky_2(double* Ah[NT][NT])
{
int i, j, k;
#pragma analysis_check assert range(k:0:256:0)
for (k = 0; k < NT; k++) {
#pragma omp task untied inout(Ah[k][k]) 
omp_potrf (Ah[k][k], TS, TS);
#pragma analysis_check assert range(i:1:256:0)
for (i = k + 1; i < NT; i++) {
#pragma omp task untied in(Ah[k][k]) inout(Ah[k][i]) 
omp_trsm (Ah[k][k], Ah[k][i], TS, TS);
}
#pragma analysis_check assert range(i:1:256:0)
for (i = k + 1; i < NT; i++) {
#pragma analysis_check assert range(j:1:256:0)
for (j = k + 1; j < i; j++) {
#pragma omp task untied in(Ah[k][i], Ah[k][j]) inout(Ah[j][i]) 
omp_gemm (Ah[k][i], Ah[k][j], Ah[j][i], TS, TS);
}
#pragma omp task untied in(Ah[k][i]) inout(Ah[i][i]) 
omp_syrk (Ah[k][i], Ah[i][i], TS, TS);
}
}
#pragma omp taskwait
}
