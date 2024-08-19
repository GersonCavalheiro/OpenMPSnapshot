#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#define THREADS 16
#define N 1000000000 
int main(int argc, char** argv)
{
unsigned long total = 0;
double start, end;
omp_set_num_threads(THREADS);
int *vector = malloc(N * sizeof(int));
for(unsigned long i = 0; i < N; i++)
vector[i] = 1;
start = omp_get_wtime();
for (unsigned long i = 2; i <= (unsigned long) sqrt(N); i++) 
{
if (vector[i] == 1)
{
#pragma omp parallel for schedule(static)
for(unsigned long j = 2 * i; j < N; j += i) 
{
#pragma omp atomic write
vector[j] = 0;	
}
}
}
#pragma omp parallel for reduction(+:total)
for (unsigned long i = 0; i < N; i++)
{
if (vector[i])
total++;
}
end = omp_get_wtime();
printf("Total de primos: %lu | Tempo total do algoritmo: %.3f segundos\n", total - 2, end - start);
return 0;
}