#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
int testConflict(int **board, int row, int col, int n) {
for (int i = 0; i < row; i++) {
if (board[i][col])
return 0;
}
for (int i = row, j = col; i >= 0 && j >= 0; i--, j--) {
if (board[i][j])
return 0;
}
for (int i = row, j = col; i >= 0 && j < n; i--, j++) {
if (board[i][j])
return 0;
}
return 1;
}
int solve(int **board, int row, int n) {
int solutions = 0;
if (row == n) {
return 1;
}
for (int col = 0; col < n; col++) {
if (testConflict(board, row, col, n)) {
board[row][col] = 1;
solutions += solve(board, row + 1, n);
board[row][col] = 0;
}
}
return solutions;
}
int main(int argc, char *argv[]) {
int n;
long int solutions;
n = atoi(argv[1]);
if (argc < 3)
omp_set_num_threads(0);
else
omp_set_num_threads(atoi(argv[2]));
double start = omp_get_wtime();
#pragma omp parallel reduction(+:solutions)
{
int **board = (int **) malloc(n * sizeof(int *));
for (int i = 0; i < n; i++) {
board[i] = (int *) malloc(n * sizeof(int));
for (int j = 0; j < n; j++) {
board[i][j] = 0;       
}
}
#pragma omp for
for (int col = 0; col < n; col++) {
board[0][col] = 1;
solutions += solve(board, 1, n);
board[0][col] = 0;
}
for (int i = 0; i < n; i++) {
free(board[i]);
}
free(board);
}
double end = omp_get_wtime();
printf("Solucoes: %ld\n", solutions);
printf("Tempo de execucao: %fs\n", end - start);
return 0;
}
