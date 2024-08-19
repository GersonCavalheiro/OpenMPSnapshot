#include <iostream>
#include <chrono>
#include <cmath>
#include <omp.h>
#include <stdlib.h>
#include <numa.h>

using namespace std;

void print_pivot(int *&, int);
void print_matrix(double **&, int);
void swap_values(int *&, int, int);
void swap_rows(double *&, double *&);
void swap_interval(double **&, int, int);
void initialize_random_matrix(double **&, double **&, int);
void lu_decomposition(double **&, double **&, double **&, int *&, int);
void initialize_for_lu_decomposition(double **&, double **&, double **&, int *&, int);

double l2_norm(double);
double verify_result(int **&, double **&, double **&, double **&, int);

void print_matrix(double **&matrix, int size)
{
for (int i = 0; i < size; ++i)
{
for (int j = 0; j < size; ++j)
cout << matrix[i][j] << " ";
cout << endl;
}
cout << endl;
}

void print_pivot(int *&P, int size)
{
for (int i = 0; i < size; ++i)
cout << P[i] << " ";

cout << endl;
}

void initialize_random_matrix(double **&matrix, double **&matrix_clone, int size)
{
matrix = new double *[size];
matrix_clone = new double *[size];
#pragma omp parallel for
for (int i = 0; i < size; ++i)
{
matrix[i] = (double *)numa_alloc_onnode(sizeof(double) * size, i < size / 2 ? 0 : 1);
matrix_clone[i] = (double *)numa_alloc_onnode(sizeof(double) * size, i < size / 2 ? 0 : 1);
}

unsigned int seed;
#pragma omp parallel private(seed)
{
seed = 221 * (omp_get_thread_num() + 1);
#pragma omp for schedule(static)
for (int i = 0; i < size; ++i)
{
for (int j = 0; j < size; ++j)
{
matrix[i][j] = matrix_clone[i][j] = (rand_r(&seed) % 10000) - 5000;
}
}
}

return;
}

void initialize_for_lu_decomposition(double **&A, double **&L, double **&U, int *&P, int size)
{
L = (double **)numa_alloc_onnode(sizeof(double *) * size, 0);
U = (double **)numa_alloc_onnode(sizeof(double *) * size, 1);
P = new int[size];
#pragma omp parallel for
for (int i = 0; i < size; ++i)
{
L[i] = (double *)numa_alloc_onnode(sizeof(double) * size, i < size / 2 ? 0 : 1);
U[i] = (double *)numa_alloc_onnode(sizeof(double) * size, i < size / 2 ? 0 : 1);
P[i] = i;
}

#pragma omp parallel for schedule(static)
for (int i = 0; i < size; ++i)
{
for (int j = 0; j < size; ++j)
{
L[i][j] = (i == j) ? 1 : 0;
U[i][j] = 0;
}
}
}

void lu_decomposition(double **&A, double **&L, double **&U, int *&P, int size)
{
initialize_for_lu_decomposition(A, L, U, P, size); 

int new_k = 0;
double current_max = 0;
for (int k = 0; k < size; ++k)
{
current_max = 0;
for (int i = k; i < size; ++i)
{
if (abs(A[i][k]) > current_max)
{
current_max = abs(A[i][k]);
new_k = i;
}
}
if (current_max == 0)
throw invalid_argument("Input is a singular matrix");

if (k != new_k)
{
swap_values(P, k, new_k);
swap_rows(A[k], A[new_k]);
swap_interval(L, k, new_k);
}

U[k][k] = A[k][k];

#pragma omp parallel for schedule(static)
for (int i = k + 1; i < size; ++i)
{
L[i][k] = A[i][k] / U[k][k];
U[k][i] = A[k][i];
}

#pragma omp parallel for schedule(static)
for (int i = k + 1; i < size; ++i)
{
for (int j = k + 1; j < size; ++j)
A[i][j] -= L[i][k] * U[k][j];
}
}

return;
}

void swap_values(int *&P, int row1, int row2)
{
int tempt = P[row2];
P[row2] = P[row1];
P[row1] = tempt;
return;
}

void swap_rows(double *&row1, double *&row2)
{
double *tempt = row2;
row2 = row1;
row1 = tempt;
return;
}

void swap_interval(double **&L, int row, int new_row)
{
for (int i = 0; i < row; ++i)
{
double tempt = L[row][i];
L[row][i] = L[new_row][i];
L[new_row][i] = tempt;
}
}

double verify_result(int *&P, double **&A, double **&L, double **&U, int size)
{
double residual = 0;

#pragma omp parallel for reduction(+ \
: residual) schedule(static)
for (int i = 0; i < size; ++i)
for (int j = 0; j < size; ++j)
{
for (int k = 0; k < size; ++k)
A[P[i]][j] -= L[i][k] * U[k][j];
residual += l2_norm(A[P[i]][j]);
}

printf("The sum of Euclidean norms is: %e\n", residual);
}

double l2_norm(double value)
{
return sqrt(pow(value, 2));
}

int main(int argc, char **argv)
{
int matrix_size = atoi(argv[1]), nworkers = atoi(argv[2]); 
int *P;
double **A, **original_matrix, **L, **U;
chrono::steady_clock clock;
omp_set_num_threads(nworkers); 
initialize_random_matrix(A, original_matrix, matrix_size);

auto start = clock.now();
lu_decomposition(A, L, U, P, matrix_size);
auto end = clock.now();

auto time_span = static_cast<chrono::duration<double>>(end - start);
printf("LU Decomposition finished in: %.4f seconds for using %d workers\n", time_span.count(), nworkers);

verify_result(P, original_matrix, L, U, matrix_size);

for (int i = 0; i < matrix_size; i++)
{
numa_free(A[i], matrix_size * sizeof(double));
numa_free(original_matrix[i], matrix_size * sizeof(double));
numa_free(L[i], matrix_size * sizeof(double));
numa_free(U[i], matrix_size * sizeof(double));
}
numa_free(L, matrix_size * sizeof(double *));
numa_free(U, matrix_size * sizeof(double *));

return 0;
}
