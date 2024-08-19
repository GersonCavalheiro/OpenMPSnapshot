#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

using namespace std;

static void get_timings();

static void free_matrix(double **matrix);

static double run_experiment();

static double **initialize_matrix(bool random);

static double **matrix_multiply_parellel(double **A, double **B, double **C);

static int n; 
static int sample_size; 


static void program_help(char *program_name) {
fprintf(stderr, "usage: %s <matrix_size> <sample_size>\n", program_name);
exit(0);
}


static void initialize(int argc, char *argv[]) {
if (argc != 3) {
program_help(argv[0]);
}

sscanf(argv[1], "%d", &n);
sscanf(argv[2], "%d", &sample_size);

if (sample_size <= 0 || n <= 0 || n > 2000) {
program_help(argv[0]);
}
}


int main(int argc, char *argv[]) {
initialize(argc, argv);
printf(
"Matrix size : %d | Sample size : %d\n\n", 
n, 
sample_size
);

get_timings();
printf("\n");

return 0;
}


void get_timings() {
double total_time = 0.0;

for (int i = 0; i < sample_size; i++) {
total_time += run_experiment();
}

double average_time = total_time / sample_size;
printf("Parallel for calculation time : %.4f seconds\n", average_time);
}


double run_experiment() {
srand(static_cast<unsigned> (time(0)));
double start, finish, elapsed;

double **A = initialize_matrix(true);
double **B = initialize_matrix(true);
double **C = initialize_matrix(false);

start = clock();
C = matrix_multiply_parellel(A, B, C);
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
