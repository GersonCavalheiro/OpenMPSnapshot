#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#ifndef N
#define N 1000
#endif
#define RUN_TEST 1
#ifndef SPARSE
#define SPARSE 0
#endif
#define CLOUD_DEVICE 0
double rtclock() {
struct timezone Tzp;
struct timeval Tp;
int stat;
stat = gettimeofday(&Tp, &Tzp);
if (stat != 0)
printf("Error return from gettimeofday: %d", stat);
return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}
void iNt_array(float *A, float *B, float *C, float *D) {
int i, j;
for (i = 0; i < N; i++) {
for (j = 0; j < N; j++) {
if ((i != j || (i%2==0)) && SPARSE) {
A[i * N + j] = 0;
} else {
A[i * N + j] = ((float)i * j) / N;
}
}
}
for (i = 0; i < N; i++) {
for (j = 0; j < N; j++) {
if ((i != j || (i%2==0)) && SPARSE) {
B[i * N + j] = 0;
} else {
B[i * N + j] = ((float)i * (j + 1)) / N;
}
}
}
for (i = 0; i < N; i++) {
for (j = 0; j < N; j++) {
if ((i != j || (i%2==0)) && SPARSE) {
D[i * N + j] = 0;
} else {
D[i * N + j] = ((float)i * (j + 2)) / N;
}
}
}
}
int compareResults(float *E, float *E_CLOUD) {
int i, j, fail;
fail = 0;
for (i = 0; i < N; i++) {
for (j = 0; j < N; j++) {
if (E[i * N + j] != E_CLOUD[i * N + j]) {
fail++;
}
}
}
printf("Non-Matching CPU-GPU Outputs: %d\n", fail);
return fail;
}
void mm2_cpu(float *A, float *B, float *C, float *D,
float *E) {
int i, j, k;
for (i = 0; i < N; i++) {
for (j = 0; j < N; j++) {
C[i * N + j] = 0.0;
for (k = 0; k < N; ++k) {
C[i * N + j] += A[i * N + k] * B[k * N + j];
}
}
}
for (i = 0; i < N; i++) {
for (j = 0; j < N; j++) {
E[i * N + j] = 0.0;
for (k = 0; k < N; ++k) {
E[i * N + j] += C[i * N + k] * D[k * N + j];
}
}
}
}
void mm2_OMP(float *A, float *B, float *C, float *D,
float *E) {
#pragma omp target map(to: A[:N*N], B[:N*N], D[:N*N]) map(tofrom: C[:N*N], E[:N*N]) device(CLOUD_DEVICE)
{
#pragma omp parallel for collapse (1)
for (int i = 0; i < N; i++) {
#pragma omp target data map (to: A[i * N : (i+1) * N]) map(tofrom: C[ i * N : (i+1) * N])
for (int j = 0; j < N; j++) {
C[i * N + j] = 0.0;
for (int k = 0; k < N; ++k) {
C[i * N + j] += A[i * N + k] * B[k * N + j];
}
}
}
#pragma omp parallel for collapse (1)
for (int i = 0; i < N; i++) {
#pragma omp target data map(to:C[i * N : (i+1) * N]) map (tofrom: E[ i * N : (i+1) * N])
for (int j = 0; j < N; j++) {
E[i * N + j] = 0.0;
for (int k = 0; k < N; ++k) {
E[i * N + j] += C[i * N + k] * D[k * N + j];
}
}
}
}
}
int main(int argc, char **argv) {
double t_start, t_end, t_start_GPU, t_end_GPU;
int fail = 0;
float *C;
float *C_CLOUD;
float *A;
float *B;
float *D;
float *E;
float *E_CLOUD;
C = (float *)malloc(N * N * sizeof(float));
C_CLOUD = (float *)malloc(N * N * sizeof(float));
A = (float *)malloc(N * N * sizeof(float));
B = (float *)malloc(N * N * sizeof(float));
D = (float *)malloc(N * N * sizeof(float));
E = (float *)malloc(N * N * sizeof(float));
E_CLOUD = (float *)malloc(N * N * sizeof(float));
fprintf(stdout,
"<< Linear Algebra: 2 Matrix Multiplications (D=A.B; E=C.D , N = %d) >>\n", N);
iNt_array(A, B, C, D);
t_start_GPU = rtclock();
mm2_OMP(A, B, C_CLOUD, D, E_CLOUD);
t_end_GPU = rtclock();
fprintf(stdout, "OMPCLOUD Runtime: %0.6lfs\n", t_end_GPU - t_start_GPU);
#ifdef RUN_TEST
t_start = rtclock();
mm2_cpu(A, B, C, D, E);
t_end = rtclock();
fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
fail += compareResults(E, E_CLOUD);
#endif
free(C);
free(A);
free(B);
free(D);
free(E);
free(E_CLOUD);
return fail;
}
