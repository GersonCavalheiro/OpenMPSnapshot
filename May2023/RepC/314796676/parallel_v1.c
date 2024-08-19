#include <stdio.h>
#include <omp.h>
#include "common.c"
#define N               20
#define ITERATION_COUNT 10
#define NUM_OF_THREADS  4
#define VERBOSE         0
#define PRINT_RESULTS   0
#define PRINT_THREADS   0
void assignValues(int matrix[N][N]);
int sumAdjacents(int matrix[N][N], int row, int column);
void playGame(int srcMatrix[N][N], int destMatrix[N][N]);
int getValue(int matrix[N][N], int i, int j);
int main(int argc, char *argv[]) {
omp_set_num_threads(NUM_OF_THREADS);
printf("Program started with %d threads and %d iterations.\n", NUM_OF_THREADS, ITERATION_COUNT);
int matrixA[N][N];
int matrixB[N][N];
double start = omp_get_wtime();
assignValues(&matrixA);
if (PRINT_RESULTS > 0) {
printMatrix("Initial Matrix", N, N, matrixA);
}
for (int iteration = 0; iteration < (ITERATION_COUNT / 2); iteration++) {
if(VERBOSE > 0) {
printf("Iteration started: %d.\n", iteration * 2 + 1);
}
playGame(matrixA, matrixB);
if(VERBOSE > 0) {
printf("Iteration started: %d.\n", (iteration + 1) * 2);
}
playGame(matrixB, matrixA);
}
logTime("Program finished. \t\t\t", start, omp_get_wtime());
}
void playGame(int srcMatrix[N][N], int destMatrix[N][N]) {
int threads_matrix[N][N];
int i, j;
#pragma omp parallel for private(j) shared(destMatrix) num_threads(NUM_OF_THREADS) schedule(static)
for (i = 0; i < N; i++) {
for (j = 0; j < N; j++) {
if (VERBOSE > 0) {
printf("Play Game\ti = %d, j = %d, threadId = %d \n", i, j, omp_get_thread_num());
}
threads_matrix[i][j] = omp_get_thread_num();
destMatrix[i][j] = getValue(srcMatrix, i, j);
}
}
#pragma omp barrier
if (PRINT_RESULTS > 0) {
printMatrix("Final Matrix :", N, N, destMatrix);
}
if (PRINT_THREADS > 0) {
printMatrix("Thread Matrix :", N, N, threads_matrix);
}
}
int getValue(int matrix[N][N], int i, int j) {
if (i == 0 || i == N - 1 || j == 0 || j == N - 1) {
return 0;
}
if (sumAdjacents(matrix, i, j) > 5) {
return 1;
} else {
return 0;
}
}
int sumAdjacents(int matrix[N][N], int row, int column) {
int sum = 0;
for (int i = row - 1; i <= row + 1; i++) {
for (int j = column - 1; j <= column + 1; j++) {
sum += matrix[i][j];
}
}
return sum;
}
void assignValues(int matrix[N][N]) {
for (int i = 0; i < N; i++) {
for (int j = 0; j < N; j++) {
matrix[i][j] = rand() % 2;
}
}
}