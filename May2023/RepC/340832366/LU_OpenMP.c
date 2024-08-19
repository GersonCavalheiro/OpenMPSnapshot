#include<stdio.h>
#include<stdlib.h>
#include<malloc.h>
#include<omp.h>
int n_threads = 1;
int **A;
int N = 0;
int **L, **U;
void read_A(const char *input_file) {
FILE* fp = fopen(input_file,"r");
int m, n;
fscanf(fp, "%d %d", &m, &n);
if (m != n) {
perror("矩阵A不是方阵，请检查输入数据！");
exit(-1);
}
N = n;
A = (int **)calloc(N, sizeof(int*));
for (int i = 0; i < N; i++) {
A[i] = (int *)calloc(N, sizeof(int));
for (int j = 0; j < N; j++) {
fscanf(fp, "%d", &A[i][j]);
}
}
L = (int **)calloc(N, sizeof(int*));
for (int i = 0; i < N; i++) {
L[i] = (int *)calloc(N, sizeof(int));
for (int j = 0; j < N; j++) {
L[i][j] = 0;
}
}
U = (int **)calloc(N, sizeof(int*));
for (int i = 0; i < N; i++) {
U[i] = (int *)calloc(N, sizeof(int));
for (int j = 0; j < N; j++) {
U[i][j] = 0;
}
}
}
int sum_i_j_K(int i, int j, int K) {
int res = 0;
for (int k = 0; k < K; k++) {
res += L[i][k] * U[k][j];
}
return res;
}
void printLU() {
FILE* fpL = fopen("L.out", "w");
FILE* fpU = fopen("U.out", "w");
for (int i = 0; i < N; i++) {
for (int j = 0; j < N; j++) {
fprintf(fpL, "%d ", L[i][j]);
fprintf(fpU, "%d ", U[i][j]);
}
fprintf(fpL, "\n");
fprintf(fpU, "\n");
}
}
int main(int argc, char *argv[]) {
if (argc >= 2) n_threads = atoi(argv[1]);
if (argc >= 3) {
read_A(argv[2]);
}
else {
read_A("LU.in");
}
omp_set_num_threads(n_threads);
double ts = omp_get_wtime();
for (int i = 0; i < N; i++) {
U[i][i] = A[i][i] - sum_i_j_K(i, i, i);
L[i][i] = 1;
#pragma omp parallel for
for (int j = i+1; j < N; j++) {
U[i][j] = A[i][j] - sum_i_j_K(i, j, i);
L[j][i] = (A[j][i] - sum_i_j_K(j, i, i)) / U[i][i];
}
}
printLU();
double te = omp_get_wtime();
printf("Time:%f s\n", te - ts);
}