#include <omp.h>
#define N 100

int main() {
int A[N];
#pragma omp for simd
for (int i = 3; i < N; i++) {
A[i] = A[i - 3] + 2;
}
return 0;
}
