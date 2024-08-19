#include <omp.h>
#define N 100

int main() {
int sum = 0;
#pragma omp ordered simd
for (int i = 0; i < N; i++) {
sum += i;
}
return sum;
}
