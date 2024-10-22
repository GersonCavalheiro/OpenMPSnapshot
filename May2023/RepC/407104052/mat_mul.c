#include "mat_mul.h"
#include "util.h"
#include <omp.h>
void mat_mul(float *A, float *B, float *C, int M, int N, int K, int num_threads) {
omp_set_num_threads(num_threads);
#pragma omp parallel
zero_mat(C, M, N);
int r = 16;
#pragma omp parallel for schedule (dynamic)
for (int i = 0; i < M; i++) {
for (int k = 0; k < K; k+= r) {
for (int j = 0; j < N; j += r) {
for (int kb = k; kb < k+r; kb++){
for (int jb = j; jb < j+r; jb++){
C[i * N + jb] += A[i*K + kb]*B[kb*N +jb];
}
}
}
}
}
}
