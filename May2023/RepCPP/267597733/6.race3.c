#include <omp.h>
#define N 20

int main() {
int i, j, A[N], a[N][N], B[N][N];
#pragma omp parallel for private(j)
for (i = 0; i < N; i++)
for (j = 0; j < i; j++) {
A[j] = a[i][j];
B[i][j] = A[j];
}
}
