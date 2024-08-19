#include <omp.h>
#define N 100

int main() {
int sum = 0;
#pragma omp single
for (int i = 0; i < N; i++) {
sum += i;
}
sum = sum / N;
return sum;
}
