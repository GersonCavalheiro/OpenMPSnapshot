#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define N 1000
int main() {
long **mat1 = (long **)malloc(N * sizeof(long *));
long **mat2 = (long **)malloc(N * sizeof(long *));
long **res = (long **)malloc(N * sizeof(long *));
#pragma omp parallel num_threads(2)
{
#pragma omp for schedule(dynamic, 8000)
for (int i = 0; i < N; i++) {
mat1[i] = (long *)malloc(N * sizeof(long));
mat2[i] = (long *)malloc(N * sizeof(long));
res[i] = (long *)malloc(N * sizeof(long));
}
#pragma omp for collapse(2) schedule(dynamic, 8000)
for (int i = 0; i < N; i++) {
for (int j =0; j < N; j++) {
mat1[i][j] = 1;
mat2[i][j] = 1;
}
}
}
#pragma omp parallel num_threads(2)
{
#pragma omp for collapse(3) schedule(dynamic, 8000)
for (int i = 0; i < N; i++) {
for (int j = 0; j < N; j++) {
for (int k = 0; k < N; k++) {
res[i][j] += mat1[i][k] * mat2[k][j]; 
}
}
}
}
}
