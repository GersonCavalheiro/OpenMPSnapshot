#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#define M 512
#define B 32
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
en = M;
start = CLOCK();
#pragma omp parallel for private(i, j, k, kk, jj, sum) shared(a, b, c, en) default(none) collapse(3) schedule(dynamic, 4)
{
for (kk = 0; kk < en; kk += B)
for (jj = 0; jj < en; jj += B)
for (i = 0; i < M; i++)
for (j = jj; j < jj + B; j++)
{
sum = c[i][j];
for (k = kk; k < kk + B; k=k+4)
{
sum += a[i][k+0] * b[k+0][j];
sum += a[i][k+1] * b[k+1][j];
sum += a[i][k+2] * b[k+2][j];
sum += a[i][k+3] * b[k+3][j];
}
c[i][j] = sum;
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
