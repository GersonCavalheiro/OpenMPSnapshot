#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

using namespace std;

static void get_timings(int algo_type, char *msg);

static void free_matrix(double **matrix);

static double run_experiment(int algo_type);

static double **initialize_matrix(bool random);

static double **matrix_multiply(double **A, double **B, double **C);

static double **matrix_multiply_parellel(double **A, double **B, double **C);


static int n; 
static int sample_size; 


int main(int argc, char **argv) {

for (int matrix_size = 200; matrix_size <= 2000; matrix_size += 200) {
n = matrix_size;
printf("Matrix size : %d\n--------------------\n", matrix_size);
fflush(stdout);

switch (n) {
case 200:
sample_size = 25;
break;
case 400:
sample_size = 20;
break;
case 600:
sample_size = 15;
break;
default:
sample_size = 10;
}

get_timings(1, (char *) "Serial");

get_timings(2, (char *) "Parallel");

printf("\n");
printf("Samples provided: %d\n\n", sample_size);

printf("\n");
fflush(stdout);
}

return 0;
}


void get_timings(int algo_type, char *msg) {
double total_time = 0.0;
double execution_times[sample_size];

for (int i = 0; i < sample_size; i++) {
double elapsed_time = run_experiment(algo_type);
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
printf("Samples required: %.4f\n\n", samples);
fflush(stdout);
}
}


double run_experiment(int algo_type) {
srand(static_cast<unsigned> (time(0)));
double start, finish, elapsed;

double **A = initialize_matrix(true);
double **B = initialize_matrix(true);
double **C = initialize_matrix(false);

start = clock();
C = algo_type == 1 ? matrix_multiply(A, B, C) : matrix_multiply_parellel(A, B, C);
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


double **matrix_multiply(double **A, double **B, double **C) {
for (int row = 0; row < n; row++) {
for (int column = 0; column < n; column++) {
for (int itr = 0; itr < n; itr++) {
C[row][column] += A[row][itr] * B[itr][column];
}
}
}

return C;
}


double **matrix_multiply_parellel(double **A, double **B, double **C) {
int row, column, itr;
#pragma omp parallel shared(A, B, C) private(row, column, itr)
{
#pragma omp for schedule(static)
for (row = 0; row < n; row++) {
for (column = 0; column < n; column++) {
for (itr = 0; itr < n; itr++) {
C[row][column] += A[row][itr] * B[itr][column];
}
}
}
}

return C;
}
