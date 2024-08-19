#include <stdio.h>
#include <omp.h>
int main() {
int n = 100;
int a[n], b[n], c[n];
int i;
for (i = 0; i < n; i++) {
a[i] = i;
b[i] = 2 * i;
}
#pragma omp parallel for schedule(guided)
for (i = 0; i < n; i++) {
c[i] = a[i] + b[i];
printf("Thread %d computed c[%d]\n", omp_get_thread_num(), i);
}
printf("c[0] = %d, c[%d] = %d\n", c[0], n-1, c[n-1]);
return 0;
}
