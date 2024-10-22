#include <omp.h>
#define N 20

int main() {
double A[N], B[N], sum0 = 0.0, sum1 = 0.0;
#pragma omp parallel
{
#pragma omp for 
for (int i = 0; i < N; i++) {
A[i] = i;
B[i] = i * i;
}
#pragma omp for reduction(+ : sum0)
for (int i = 0; i < N; i++) {
sum0 += A[i] * B[i];
}
}

for (int i = 0; i < N; i++) {
sum1 += i * i * i;
}

return (sum1 - sum0);
}
