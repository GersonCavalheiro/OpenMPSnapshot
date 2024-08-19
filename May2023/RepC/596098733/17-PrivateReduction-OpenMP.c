#include <stdio.h>
#include <omp.h>
#define N 1000
int main() {
int i = 0;
int a[N], sum = 0;
for (int i = 0; i < N; i++) {
a[i] = i + 1;
}
#pragma omp parallel for private(i) reduction(+:sum)
for (i = 0; i < N; i++) {
sum += a[i];
}
printf("The sum of the array is: %d\n", sum);
return 0;
}
