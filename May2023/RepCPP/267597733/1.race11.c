#include <omp.h>
#define N 20

int main() {
int A[N][N];
#pragma omp parallel
{
#pragma omp for
for (int i = 1; i < N; i++)
for (int j = 1; j < N; j++)
A[i][j] = A[i][j - 1];

#pragma omp for
for (int i = 1; i < N; i++)
for (int j = 1; j < N; j++)
A[i][j] = A[i - 1][j];

#pragma omp for
for (int i = 1; i < N; i++)
for (int j = 1; j < N; j++)
A[i][j] = A[i][j - 1];

#pragma omp for
for (int i = 1; i < N; i++)
for (int j = 1; j < N; j++)
A[i][j] = A[i][j];
}
}
