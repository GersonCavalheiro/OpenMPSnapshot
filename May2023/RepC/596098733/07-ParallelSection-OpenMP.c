#include <stdio.h>
#include <omp.h>
#include <x86intrin.h>	
#define N 50000
static unsigned long long start, end;
int main(int argc, char *argv[]) {
int i, j;
int a[N], b[N], c[N];
srand(time(NULL));
for (i = 0; i < N; i++) {
a[i] = rand() % 10;
b[i] = rand() % 10;
}
start = __rdtsc();
#pragma omp parallel sections num_threads(4)
{
#pragma omp section
{
for (i = 0; i < N; i++) {
c[i] = a[i] + b[i];
}
printf("He terminado suma\n");
}
#pragma omp section
{
for (j = 0; j < N; j++) {
c[j] = a[j] * b[j];
}
printf("He terminado multiplicacin\n");
}
}
end = __rdtsc();
printf("Parallel CPU time in ticks: \t\t%14llu\n", (end - start));
start = __rdtsc();
for (i = 0; i < N; i++) {
c[i] = a[i] + b[i];
}
for (j = 0; j < N; j++) {
c[j] = a[j] * b[j];
}
end = __rdtsc();
printf("Serial CPU time in ticks: \t\t%14llu\n", (end - start));
return 0;
}
