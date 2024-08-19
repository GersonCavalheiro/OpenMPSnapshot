#include <stdio.h>
#include <omp.h>
int main() {
const int N = 1000;
int a[N], b[N], c[N];
for (int i = 0; i < N; i++) {
a[i] = i;
b[i] = 2 * i;
}
#pragma omp parallel
{
#pragma omp single
{
#pragma omp taskloop
for (int i = 0; i < N; i++) {
c[i] = a[i] + b[i];
}
}
}
printf("Result:\n");
for (int i = 0; i < N; i++) {
printf("%d ", c[i]);
}
printf("\n");
return 0;
}
