#include <omp.h>
#define N 20

int main() {
int sum = 0;
for (int i = 0; i < N; i++) {
#pragma omp critical
sum += i;
}
return sum;
}
