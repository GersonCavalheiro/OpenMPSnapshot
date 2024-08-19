#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define N 10000
int main() {
long **mat = (long **)malloc(N * sizeof(long *));
long *vec = (long *)malloc(N * sizeof(long));
#pragma omp parallel num_threads(2)
{
#pragma omp for
for (int i = 0; i < N; i++) {
mat[i] = (long *)malloc(N * sizeof(long));
}
#pragma omp for collapse(2) schedule(dynamic, 8000) 
for (int i=0; i<N; i++) {
for (int j=0; j<N; j++) {
mat[i][j] = 1;
}
}
}
#pragma omp parallel num_threads(2) 
{
#pragma omp for collapse(2) schedule(dynamic, 8000)
for (int j=0; j<N; j++) {
for(int i=0; i<N; i++) {
vec [i] += mat[i][j];
}
}
}
}