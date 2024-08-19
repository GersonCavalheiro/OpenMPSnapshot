#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>		
#include <time.h>
#include <omp.h>
#include <x86intrin.h>	
#define N 10000
static unsigned long long start, end;
int main(int argc, char *argv[]) {
int i, j, num_threads, tid;
int** A = (int**)malloc(N * sizeof(int*));
int** B = (int**)malloc(N * sizeof(int*));
for (i = 0; i < N; i++) {
A[i] = (int*)malloc(N * sizeof(int));
B[i] = (int*)malloc(N * sizeof(int));
}
srand(time(NULL));
for (i = 0; i < N; i++) {
for (j = 0; j < N; j++) {
A[i][j] = rand() % 10;
}
}
#pragma omp parallel
{
#pragma omp single
num_threads = omp_get_num_threads();
}
printf("Number of cores: %d\n", num_threads);
start = __rdtsc();
#pragma omp parallel for private(i, j, tid) num_threads(num_threads)
for (i = 0; i < N; i++) {
for (j = 0; j < N; j++) {
B[i][j] = A[j][i];
}
}
end = __rdtsc();
printf("Parallel CPU time in ticks: \t\t%14llu\n", (end - start));
start = __rdtsc();
for (i = 0; i < N; i++) {
for (j = 0; j < N; j++) {
B[i][j] = A[j][i];
}
}
end = __rdtsc();
printf("Serial CPU time in ticks: \t\t%14llu\n", (end - start));
for (i = 0; i < N; i++) {
free(A[i]);
free(B[i]);
}
free(A);
free(B);
return 0;
}
