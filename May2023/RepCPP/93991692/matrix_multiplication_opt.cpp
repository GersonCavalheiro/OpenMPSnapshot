#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>


extern "C"
{
#include <immintrin.h>
}


using namespace std;

static void get_timings(char *msg);

static void free_matrix(double **matrix);

static double run_experiment();

static double **initialize_matrix(bool random);

static double **matrix_multiply_parallel_optimized(double **A, double **B, double **C);

static int n; 
static int sample_size; 


int main(int argc, char **argv) {

for (int matrix_size = 200; matrix_size <= 2000; matrix_size += 200) {
n = matrix_size;
printf("Matrix size : %d\n--------------------\n", matrix_size);
fflush(stdout);

switch (n) {
case 200:
sample_size = 200;
break;
case 400:
sample_size = 100;
break;
case 600:
sample_size = 50;
break;
default:
sample_size = 20;
}

get_timings((char *) "Optimised Parallel");
printf("Samples provided: %d\n\n", sample_size);

printf("\n");
fflush(stdout);
}

return 0;
}


void get_timings(char *msg) {
double total_time = 0.0;
double execution_times[sample_size];

for (int i = 0; i < sample_size; i++) {
double elapsed_time = run_experiment();
execution_times[i] = elapsed_time;
total_time += elapsed_time;
}

double average_time = total_time / sample_size;
printf("%s time : %.4f seconds\n", msg, average_time);
fflush(stdout);

if (sample_size > 1) {
double variance = 0.0;

for (int i = 0; i < sample_size; i++) {
variance += pow(execution_times[i] - average_time, 2);
}

double standard_deviation = sqrt(variance / (sample_size - 1));
printf("%s deviation = %.4f seconds\n", msg, standard_deviation);
fflush(stdout);

double samples =
pow((100 * 1.96 * standard_deviation) / (5 * average_time), 2);
printf("Samples required: %.4f\n", samples);
fflush(stdout);
}
}


double run_experiment() {
srand(static_cast<unsigned> (time(0)));
double start, finish, elapsed;

double **A = initialize_matrix(true);
double **B = initialize_matrix(true);
double **C = initialize_matrix(false);

start = clock();
C = matrix_multiply_parallel_optimized(A, B, C);
finish = clock();

elapsed = (finish - start) / CLOCKS_PER_SEC;

free_matrix(A);
free_matrix(B);
free_matrix(C);

return elapsed;
}


void free_matrix(double **matrix) {
for (int i = 0; i < n; i++) {
delete [] matrix[i];
}
delete [] matrix;
}


double **initialize_matrix(bool random) {
double **matrix = new double*[n];
for (int i = 0; i < n; i++)
matrix[i] = new double[n];

for (int row = 0; row < n; row++) {
for (int column = 0; column < n; column++) {
matrix[row][column] = random ? ((double)rand()/(double)(RAND_MAX/10000)) : 0.0;
}
}

return matrix;
}


double **matrix_multiply_parallel_optimized(double **A, double **B, double **C) {
int row, column, itr;
double *row_A, *row_C, *row_B;
double val_A;
#pragma omp parallel shared(A, B, C) private(row, column, itr, row_A, row_C, row_B, val_A)
{
#pragma omp for schedule(static)
for (row = 0; row < n; row++) {
row_A = A[row];
row_C = C[row];
for (itr = 0; itr < n; itr++) {
row_B = B[itr];
val_A = row_A[itr];
for (column = 0; column < n; column += 5) {
row_C[column] += val_A * row_B[column];
row_C[column + 1] += val_A * row_B[column + 1];
row_C[column + 2] += val_A * row_B[column + 2];
row_C[column + 3] += val_A * row_B[column + 3];
row_C[column + 4] += val_A * row_B[column + 4];
}
}
}
}
return C;
}
