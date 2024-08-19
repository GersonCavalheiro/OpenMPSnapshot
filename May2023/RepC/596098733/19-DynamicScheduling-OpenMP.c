#include <stdio.h>
#include <omp.h>
int main() {
int i, j, n = 100;
int a[n], b[n], c[n];
for (i = 0; i < n; i++) {
a[i] = i;
b[i] = i;
}
#pragma omp parallel for schedule(dynamic)
for (j = 0; j < n; j++) {
c[j] = a[j] + b[j];
printf("Thread %d is working on index %d\n", omp_get_thread_num(), j);
}
printf("Result:\n");
for (i = 0; i < n; i++) {
printf("%d\n", c[i]);
}
return 0;
}
