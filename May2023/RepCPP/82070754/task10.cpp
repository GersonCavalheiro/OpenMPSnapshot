#include "iostream"
#include "omp.h"
#include "../../array_utils.h"

const int N = 6;
const int M = 8;

int main() {

srand(time(NULL));

int **matrix = new int *[N];
for (int i = 0; i < N; i++) {
matrix[i] = new int[M];
for (int j = 0; j < M; j++) {
matrix[i][j] = rand() % 100;
}
}

print_matrix(matrix, N, M);
printf("\n");

int max = 0;
int min = 10;

#pragma omp parallel for num_threads(8)
for (int i = 0; i < N; i++) {
for (int j = 0; j < M; j++) {
#pragma omp critical
if (matrix[i][j] > max) {
max = matrix[i][j];
}
}
for (int j = 0; j < M; j++) {
#pragma omp critical
if (matrix[i][j] < min) {
min = matrix[i][j];
}
}
}
printf("Max is : %d, min is : %d\n", max, min);
}
