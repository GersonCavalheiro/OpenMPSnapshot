#include <omp.h>
#define N 100

int main() {
int sum = 0;
#pragma omp ordered
for (int i = 0; i < N; i++) {
sum += i;
}
return 0;
}
