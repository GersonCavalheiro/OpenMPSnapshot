#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
int n_threads = 1;
int n = 1000;
int **A, **B, **C;
void init() {
A = (int **)calloc(n, sizeof(int *));
for (int i = 0; i < n; i++) {
A[i] = (int *)calloc(n, sizeof(int));
for (int j = 0; j < n; j++) {
A[i][j] = i + 1 + j;
}
}
B = (int **)calloc(n, sizeof(int *));
for (int i = 0; i < n; i++) {
B[i] = (int *)calloc(n, sizeof(int));
for (int j = 0; j < n; j++) {
B[i][j] = 1;
}
}
C = (int **)calloc(n, sizeof(int *));
for (int i = 0; i < n; i++) {
C[i] = (int *)calloc(n, sizeof(int));
for (int j = 0; j < n; j++) {
C[i][j] = 0;
}
}
}
unsigned long long sum_C()
{
unsigned long long sum = 0;
for (int i = 0; i < n; i++)
{
for (int j = 0; j < n; j++)
{
sum += C[i][j];
}
}
return sum;
}
int main(int argc, char *argv[]) {
if (argc >= 2) n_threads = atoi(argv[1]);
if (argc >= 3) n = atoi(argv[2]);
init();
omp_set_num_threads(n_threads);
double ts = omp_get_wtime();
#pragma omp parallel for
for (int i = 0; i < n; i++) {
for (int j = 0; j < n; j++) {
for (int k = 0; k < n; k++) {
C[i][j] += A[i][k] * B[k][j];
}
}
}
printf("Sum of Matrix C:%llu\n", sum_C());
double te = omp_get_wtime();
printf("Time:%f s\n", te - ts);
}