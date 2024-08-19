#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include "timer.h"
unsigned char ** cleanBoard(unsigned int rows, unsigned int columns)
{
unsigned char **board;
unsigned int i;
board = malloc(rows * sizeof(*board));
for (i = 0; i < rows; ++i) {
board[i] = malloc(columns * sizeof(**board));
memset(board[i], ' ', columns);
}
return board;
}
unsigned char ** randomBoard(unsigned int rows, unsigned int columns)
{
unsigned char **board;
unsigned int i, j;
board = cleanBoard(rows, columns);
for (i = 0; i < rows; ++i) {
for (j = 0; j < columns; ++j) {
board[i][j] = rand() % 2 ? ' ' : 'x';
}
}
return board;
}
void deleteBoard(unsigned char **board, unsigned int rows, unsigned int columns)
{
unsigned int i;
for (i = 0; i < rows; ++i) {
free(board[i]);
}
free(board);
}
void printBoard(unsigned char **board, unsigned int rows, unsigned int columns, FILE* stream)
{
unsigned int i, j;
for (i = 0; i < rows; ++i) {
for (j = 0; j < columns; ++j) {
fprintf(stream, "%c", board[i][j]);
}
fprintf(stream, "\n");
}
}
unsigned int countNeighbors(unsigned char **board, unsigned int rows, unsigned int columns, unsigned int row, unsigned int column)
{
unsigned int count = 0;
if (row > 0 && column > 0 && board[row - 1][column - 1] == 'x') count++;
if (row > 0 && board[row - 1][column] == 'x') count++;
if (row > 0 && column < (columns - 1) && board[row - 1][column + 1] == 'x') count ++;
if (column > 0 && board[row][column - 1] == 'x') count++;
if (column < (columns - 1) && board[row][column + 1] == 'x') count++;
if (row < (rows - 1) && column > 0 && board[row + 1][column - 1] == 'x') count++;
if (row < (rows - 1) && board[row + 1][column] == 'x') count++;
if (row < (rows - 1) && column < (columns - 1) && board[row + 1][column + 1] == 'x') count++;
return count;
}
int main(int argc, char const *argv[])
{
int numThreads;
unsigned int size;
unsigned int iterations;
unsigned char **board;
unsigned char **auxBoard;
unsigned int h, i, j;
double tempo, fim, inicio;
if (argc < 4) {
printf("Error missing command line argument.\n");
return 1;
}
size = atoi(argv[1]);
iterations = atoi(argv[2]);
numThreads = atoi(argv[3]);
board = randomBoard(size, size);
auxBoard = cleanBoard(size, size);
GET_TIME(inicio);
#pragma omp parallel num_threads(numThreads) private(h,i,j)
for (h = 0; h < iterations; ++h) {
#pragma omp for private(h,i,j)
for (i = 0; i < size; ++i) {
for (j = 0; j < size; ++j) {
unsigned int neighbors = countNeighbors(board, size, size, i, j);
if (board[i][j] == ' ') {
if (neighbors == 3) {
auxBoard[i][j] = 'x';
}
else {
auxBoard[i][j] = ' ';
}
}
else {
if (neighbors < 2 || neighbors > 3) {
auxBoard[i][j] = ' ';
}
else {
auxBoard[i][j] = 'x';
}
}
}
}
#pragma omp for private(h,i,j)
for (i = 0; i < size; ++i) {
for (j = 0; j < size; ++j) {
board[i][j] = auxBoard[i][j];
}
}
}
GET_TIME(fim);
tempo = fim - inicio;
printf("Tempo: %.8lf\n", tempo);
deleteBoard(board, size, size);
deleteBoard(auxBoard, size, size);
return 0;
}