#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <unistd.h>
#define N               8
#define ITERATION_COUNT 10
#define NUM_OF_THREADS  4
#define VERBOSE         1
#define PRINT_RESULTS   1
#define PRINT_THREADS   0
#define CHUNK_SIZE      3
void assignValues(int matrix[N][N]);
void resetThreadsMatrix(int matrix[N][N]);
void printMatrix(char *id, int rowCount, int columnCount, int matrix[rowCount][columnCount]);
void logTime(char *id, double start, double end);
int main(int argc, char *argv[]) {
omp_set_num_threads(NUM_OF_THREADS);
printf("Program started with %d threads and %d iterations.\n", NUM_OF_THREADS, ITERATION_COUNT);
int matrix[N][N];
assignValues(&matrix);
if (PRINT_RESULTS > 0) {
printMatrix("Initial Matrix", N, N, matrix);
}
int threads[N][N];
resetThreadsMatrix(threads);
double start = omp_get_wtime();
#pragma omp parallel for schedule(static) num_threads(NUM_OF_THREADS)
for (int i = 0; i < N; i++) {
for (int j = i; j < N; j++) {
threads[i][j] = omp_get_thread_num();
sleep(i);
}
}
logTime("Program finished. \t\t\t", start, omp_get_wtime());
if (PRINT_RESULTS > 0) {
printMatrix("Threads Matrix (Static)", N, N, threads);
}
resetThreadsMatrix(threads);
start = omp_get_wtime();
#pragma omp parallel for schedule(dynamic) num_threads(NUM_OF_THREADS)
for (int i = 0; i < N; i++) {
for (int j = i; j < N; j++) {
threads[i][j] = omp_get_thread_num();
sleep(i);
}
}
logTime("Program finished. \t\t\t", start, omp_get_wtime());
if (PRINT_RESULTS > 0) {
printMatrix("Threads Matrix (Dynamic)", N, N, threads);
}
resetThreadsMatrix(threads);
start = omp_get_wtime();
#pragma omp parallel for schedule(dynamic, CHUNK_SIZE) num_threads(NUM_OF_THREADS)
for (int i = 0; i < N; i++) {
for (int j = i; j < N; j++) {
threads[i][j] = omp_get_thread_num();
sleep(i);
}
}
logTime("Program finished. \t\t\t", start, omp_get_wtime());
if (PRINT_RESULTS > 0) {
printMatrix("Threads Matrix (Dynamic) with chunk size", N, N, threads);
}
resetThreadsMatrix(threads);
start = omp_get_wtime();
#pragma omp parallel for schedule(guided) num_threads(NUM_OF_THREADS)
for (int i = 0; i < N; i++) {
for (int j = i; j < N; j++) {
threads[i][j] = omp_get_thread_num();
sleep(i);
}
}
logTime("Program finished. \t\t\t", start, omp_get_wtime());
if (PRINT_RESULTS > 0) {
printMatrix("Threads Matrix (Guided)", N, N, threads);
}
resetThreadsMatrix(threads);
start = omp_get_wtime();
#pragma omp parallel for schedule(guided, CHUNK_SIZE) num_threads(NUM_OF_THREADS)
for (int i = 0; i < N; i++) {
for (int j = i; j < N; j++) {
threads[i][j] = omp_get_thread_num();
sleep(i);
}
}
logTime("Program finished. \t\t\t", start, omp_get_wtime());
if (PRINT_RESULTS > 0) {
printMatrix("Threads Matrix (Guided) with chunk size", N, N, threads);
}
return 0;
}
void assignValues(int matrix[N][N]) {
for (int i = 0; i < N; i++) {
for (int j = 0; j < N; j++) {
matrix[i][j] = rand() % 2;
}
}
}
void printMatrix(char *id, int rowCount, int columnCount, int matrix[rowCount][columnCount]) {
char *s = (char *) malloc(1000 * sizeof(char));
sprintf(s, "%s\n", id);
for (int row = 0; row < rowCount; row++) {
sprintf(s, "%s\t\t\t\t\t", s);
for (int columns = 0; columns < columnCount; columns++) {
sprintf(s, "%s%d\t", s, matrix[row][columns]);
}
sprintf(s, "%s\n", s);
}
printf(s);
free(s);
}
void resetThreadsMatrix(int matrix[N][N]) {
for (int i = 0; i < N; i++) {
for (int j = 0; j < N; j++) {
matrix[i][j] = -1;
}
}
}
void logTime(char *id, double start, double end) {
char *s = (char *) malloc(1000 * sizeof(char));
sprintf(s, "%s", id);
sprintf(s, "%sTotal execution time : %f\n", s, end - start);
printf(s);
free(s);
}