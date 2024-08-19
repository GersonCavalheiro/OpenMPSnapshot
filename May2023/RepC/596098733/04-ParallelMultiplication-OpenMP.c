#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>		
#include <omp.h>
#include <time.h>
#include <x86intrin.h>	
#define N 100
static unsigned long long start, end;
int main(int argc, char* argv[])
{
int i, j, k;
int a[N][N];
int b[N][N];
int c[N][N];
int num_threads = omp_get_num_procs();
srand(time(NULL));
for (i = 0; i < N; i++) {
for (j = 0; j < N; j++) {
a[i][j] = rand() % 10;
b[i][j] = rand() % 10;
}
}
start = __rdtsc();
#pragma omp parallel for num_threads(num_threads) private(i, j, k)
for (i = 0; i < N; i++) {
for (j = 0; j < N; j++) {
c[i][j] = 0;
for (k = 0; k < N; k++) {
c[i][j] += a[i][k] * b[k][j];
}
}
}
end = __rdtsc();
printf("Parallel CPU time in ticks: \t\t%14llu\n", (end - start));
start = __rdtsc();
for (i = 0; i < N; i++) {
for (j = 0; j < N; j++) {
c[i][j] = 0;
for (k = 0; k < N; k++) {
c[i][j] += a[i][k] * b[k][j];
}
}
}
end = __rdtsc();
printf("Serial CPU time in ticks: \t\t%14llu\n", (end - start));
return 0;
}
