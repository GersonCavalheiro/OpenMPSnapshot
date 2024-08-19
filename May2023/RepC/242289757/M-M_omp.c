#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#define M 512
#define NUM_THREADS 12
double CLOCK()
{
struct timespec t;
clock_gettime(CLOCK_MONOTONIC, &t);
return (t.tv_sec * 1000) + (t.tv_nsec * 1e-6);
}
int main(int argc, char **argv)
{
omp_set_num_threads(NUM_THREADS);
int i, j, k, jj, kk, en;
double start, finish, total, sum;
float a[M][M], b[M][M], c[M][M];
srand(145);
for (i = 0; i < M; i++)
for (j = 0; j < M; j++)
a[i][j] = (float)rand() / (float)RAND_MAX;
for (i = 0; i < M; i++)
for (j = 0; j < M; j++)
b[i][j] = (float)rand() / (float)RAND_MAX;
for (i = 0; i < M; i++)
for (j = 0; j < M; j++)
c[i][j] = 0.;
start = CLOCK();
double x = 0;
#pragma omp parallel for private(x, i, j, k)
{
for (i = 0; i < M; i++)
{
for (j = 0; j < M; j++)
{
x = 0;
for (k = 0; k < M; k++)
{
x += a[i][k] * b[k][j];
}
c[i][j] = x;
}
}
}
finish = CLOCK();
total = finish - start;
printf("Time for the loop = %4.2f milliseconds\n", total);
printf("Element %d,%d = %f\n", 0, 0, c[0][0]);
printf("Element %d,%d = %f\n", 63, 63, c[63][63]);
printf("Element %d,%d = %f\n", 511, 511, c[511][511]);
return 0;
}
