#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef USE_EXPENSIVE_LOOP
#include <math.h>
#endif 
int main() {
size_t n = 100000000;
double *x = malloc(n * sizeof *x);
double *y = malloc(n * sizeof *y);
double *z = malloc(n * sizeof *z);
double alpha = 2.23;
double start_time = omp_get_wtime();
#pragma omp parallel
{
#pragma omp for
for (int i = 0; i < n; ++i) {
x[i] = 1.23 * i;
y[i] = 4.56 * i;
}
#pragma omp barrier
#pragma omp for
for (int i = 0; i < n; ++i) {
#ifndef USE_EXPENSIVE_LOOP
z[i] = alpha * x[i] + y[i];
#else
z[i] = log(exp(alpha * x[i]) * exp(y[i]));
#endif 
}
}
double time = omp_get_wtime() - start_time;
printf("Max threads:  %3d\n", omp_get_max_threads());
printf("Elapsed time: %.3es\n", time);
free(x);
free(y);
free(z);
}
