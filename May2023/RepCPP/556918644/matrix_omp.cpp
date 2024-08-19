
#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include <omp.h>

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

#define THREADS 12

void matrixMult(int *a, int *b, int *c, int N)
{
int temp_sum;

#pragma omp parallel for collapse(2) num_threads(THREADS)
for (int i = 0; i < N; i++)
{
for (int j = 0; j < N; j++)
{
temp_sum = 0;
for (int k = 0; k < N; k++)
temp_sum += a[i * N + k] * b[k * N + j]; 
c[i * N + j] = temp_sum;
}
}
}

void initMatrix(int *a, int N)
{
for (int i = 0; i < N; i++)
for (int j = 0; j < N; j++)
a[i * N + j] = rand() % 100;
}

void printMatrix(int *m, int N)
{
printf("\n");

for (int i = 0; i < N; i++)
{
for (int j = 0; j < N; j++)
printf("%d ", m[i * N + j]);
printf("\n");
}
}

int main(int argc, char **argv)
{
int N = 1 << atoi(argv[1]); 

size_t bytes = N * N * sizeof(int);

int *h_a, *h_b, *h_c;

h_a = (int *)malloc(bytes);
h_b = (int *)malloc(bytes);
h_c = (int *)malloc(bytes);

int *d_a, *d_b, *d_c;

initMatrix(h_a, N);
initMatrix(h_b, N);


auto start = high_resolution_clock::now();

matrixMult(h_a, h_b, h_c, N);

auto stop = high_resolution_clock::now();



duration<double, std::milli> ms_double = stop - start;

printf("\nCompleted successfully!\n");
printf("matrixMult() execution time on the CPU: %f ms\n", ms_double.count());

free(h_a);
free(h_b);

return 0;
}