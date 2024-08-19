#include <omp.h>
#define N 100

int main() {
int A[N];
#pragma omp for
for (int i = 1; i < N; i++) {
A[i] = A[i] + A[i - 1];
}
return 0;
}
