#include "matrix.h"
#include <omp.h>
void multiply(int N, unsigned long A[][2048], unsigned long B[][2048], unsigned long C[][2048]) {
for (int i = 0; i < N; i++)
for (int j = i; j < N; j++)
{
unsigned long tmp = B[i][j];
B[i][j] = B[j][i];
B[j][i] = tmp;
}
omp_set_num_threads(6);
#pragma omp parallel for
for (int i = 0; i < N; i++) {
for (int j = 0; j < N; j++) {
unsigned long sum = 0;    
for (int k = 0; k < N; k++)
sum += A[i][k] * B[j][k];
C[i][j] = sum;
}
}
}