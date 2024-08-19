#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <x86intrin.h>	
#define N 200000000
static unsigned long long start, end;
int main() {
int i;
double *a = (double*)malloc(N * sizeof(double));
int num_threads = omp_get_num_threads();
for (i = 0; i < N; i++) {
a[i] = i * 0.5;
}
start = __rdtsc();
#pragma omp parallel for num_threads(6)
for (i = 0; i < N; i++) {
a[i] = a[i] * a[i];
}
end = __rdtsc();
printf("Result: %lf\n", a[N-1]);
printf("Parallel CPU time in ticks: \t\t%14llu\n", (end - start));
for (i = 0; i < N; i++) {
a[i] = i * 0.5;
}
start = __rdtsc();
for (i = 0; i < N; i++) {
a[i] = a[i] * a[i];
}
end = __rdtsc();
printf("Result: %lf\n", a[N-1]);
printf("Serial CPU time in ticks: \t\t%14llu\n", (end - start));
free(a);
return 0;
}
