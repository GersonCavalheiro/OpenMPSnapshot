#include <omp.h>
#define M 200
#define N 200

int main() {
double A[M], B[M][N], C[N], sum0 = 0.0;
for (int i = 0; i < M; i++) {
#pragma omp parallel for 
for (int j = 0; j < N; j++) {
sum0 += B[i][j] * C[j];
}
A[i] = sum0;
}
}
