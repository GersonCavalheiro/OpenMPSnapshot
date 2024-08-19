#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
int main () {
size_t n = 1000000;
int x[n];
#pragma omp parallel
{
int id = omp_get_thread_num();
if (id == 0) {
printf("Executing parallel region...\n");
} else {
#pragma omp for
for (int i = 0; i < n; i++) {
x[i] = i;
}
}
}
}
