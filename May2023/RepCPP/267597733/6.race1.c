#include <omp.h>
#define N 20

int main() {
int x[N], b[N], L[N][N];
#pragma omp parallel for
for (int i = 0; i < N; i++) {
x[i] = b[i];
for (int j = 0; j < i; j++)
x[i] = x[i] - L[i][j] * x[j];
x[i] = x[i] / L[i][i];
}
}
