#include <omp.h>
#define M 200
#define N 200

double sum = 0.0;
#pragma omp threadprivate(sum)

int main() {
double A[M], B[M][N], C[N];
#pragma omp parallel for
for (int i = 0; i < M; i++) {
for (int j = 0; j < N; j++) {
sum += B[i][j] * C[j];
}
A[i] = sum;
sum = 0.0;
}
}
