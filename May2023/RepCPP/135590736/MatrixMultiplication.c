#include "stdio.h"
#include "math.h"
#include "stdlib.h"
#include "string.h"
#include "limits.h"
#include "sys/time.h"
#include "omp.h"

#define BLOCK_SIZE 16
#define MATRIX_SIZE 256

float A[MATRIX_SIZE][MATRIX_SIZE],
B[MATRIX_SIZE][MATRIX_SIZE],
C[MATRIX_SIZE][MATRIX_SIZE];

int min(int a, int b)
{
return a < b ? a : b;
}

int main(int argc, char*  argv[])
{
struct timeval start;
struct timeval end;
double elapsedTime;
double numOps;
float gFLOPS;

for (int i = 0; i < MATRIX_SIZE; ++i)
for (int k = 0; k < MATRIX_SIZE; ++k)
A[i][k] = B[i][k] = 1.0;
memset(C, 0, sizeof(C[0][0] * MATRIX_SIZE * MATRIX_SIZE));

int k, j, i, jj, kk;

gettimeofday(&start, NULL);

for (k = 0; k < MATRIX_SIZE; k += BLOCK_SIZE)
for (j = 0; j < MATRIX_SIZE; j += BLOCK_SIZE)
#pragma omp parallel for collapse(3)
for (i = 0; i < MATRIX_SIZE; ++i)
for (jj = j; jj < min(j + BLOCK_SIZE, MATRIX_SIZE); ++jj)
for (kk = k; kk < min(k + BLOCK_SIZE, MATRIX_SIZE); ++kk)
C[i][jj] += A[i][kk] * B[kk][jj];

gettimeofday(&end, NULL);

elapsedTime = (end.tv_sec - start.tv_sec) * 1000.0;
elapsedTime += (end.tv_usec - start.tv_usec) / 1000.0;
elapsedTime /= 1000;

numOps = 2 * pow(MATRIX_SIZE, 3);
gFLOPS = (float)(1.0e-9 * numOps / elapsedTime);

printf("Multi Core CPU  : %.3f seconds ( %f GFLOPS )\n", elapsedTime,gFLOPS);

return 0;
}