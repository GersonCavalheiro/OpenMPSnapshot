#include <omp.h>
#define N 100

int main() {
int A[N], B[N];
#pragma omp parallel shared(A) num_threads(2)
{
#pragma omp sections 
{
for (int i = 0; i < N; i++) {
A[i] = i;
}
#pragma omp section
for (int i = 0; i < N; i++) {
B[i] = i * i;
}
}
#pragma omp for
for (int i = 0; i < N; i++) {
A[i] *= B[i];
}
}
return 0;
}
