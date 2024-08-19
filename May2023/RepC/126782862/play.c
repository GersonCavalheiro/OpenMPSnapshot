#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <omp.h>
#include "game-of-life.h"
void play_in_serial (char *board, char *newboard, int rows, int cols) {
int i, j, a;
for (i = 0; i < rows; i++)
for (j = 0; j < cols; j++) {
a = adjacent_to (board, i, j, rows, cols);
if (a == 2) newboard[i * cols + j] = board[i * cols + j];
if (a == 3) newboard[i * cols + j] = 1;
if (a < 2) newboard[i * cols + j] = 0;
if (a > 3) newboard[i * cols + j] = 0;
}
for (i = 0; i < rows; i++)
for (j = 0; j < cols; j++) {
board[i * cols + j] = newboard[i * cols + j];
}
}
void play_in_parallel (char *board, char *newboard, int rows, int cols) {
int i, j, a;
#pragma omp parallel for private(i, j, a)
for (i = 0; i < rows; i++)
for (j = 0; j < cols; j++) {
a = adjacent_to (board, i, j, rows, cols);
if (a == 2) newboard[i * cols + j] = board[i * cols + j];
if (a == 3) newboard[i * cols + j] = 1;
if (a < 2) newboard[i * cols + j] = 0;
if (a > 3) newboard[i * cols + j] = 0;
}
for (i = 0; i < rows; i++)
for (j = 0; j < cols; j++) {
board[i * cols + j] = newboard[i * cols + j];
}
}