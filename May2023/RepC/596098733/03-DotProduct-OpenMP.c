#include <stdio.h>
#include <omp.h>
#include "rdtsc.h"
#define N 200000
static unsigned long long start, end;
int main() {
int i;
int a[N], b[N];
int result = 0;
for (i = 0; i < N; i++) {
a[i] = i;
b[i] = N - i;
}
start = rdtsc();
for (i = 0; i < N; i++) {
result += a[i] * b[i];
}
end = rdtsc();
printf("Serial CPU time in ticks: \t\t%llu\n",(end - start));
printf("Result: %d\n", result);
int num_threads= omp_get_thread_num();
omp_set_num_threads(num_threads);
result=0;
start = rdtsc();
#pragma omp parallel for reduction(+:result)
for (i = 0; i < N; i++) {
result += a[i] * b[i];
}
end = rdtsc();
printf("Parallel CPU time in ticks: \t\t%llu\n",(end - start));
printf("Result: %d\n", result);
return 0;
}
