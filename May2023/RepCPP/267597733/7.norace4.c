#include <omp.h>
#define M 20
#define N 20

int main() {
double data[M][N], mean[N];
#pragma omp parallel for
for (int i = 0; i < N; i++)
for (int j = 0; j < M; j++) {
data[i][j] -= mean[j];
}
}
