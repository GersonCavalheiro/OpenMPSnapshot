#include <omp.h>
#define M 20
#define N 20

int main() {
double A[M][N], B[M][N], C[N][N];
#pragma omp parallel for
for (int i = 0; i < M; i++)
for (int j = 0; j < N; j++) {
A[i][j] = 0.0;
for (int k = 0; k < N; ++k)
A[i][j] += B[i][k] * C[k][j];
}
}
