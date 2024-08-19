#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N 5000
#define EPSILON 1E-7
#define TAU 1E-5
#define MAX_ITERATION_COUNT 1000000
void generate_A(double* A, int size);
void generate_x(double* x, int size);
void generate_b(double* b, int size);
double calc_norm_square(const double* vector, int size);
void calc_Axb(const double* A, const double* x, const double* b, double* Axb, int size);
void calc_next_x(const double* Axb, double* x, double tau, int size);
int main(int argc, char **argv)
{
int iter_count;
double accuracy = EPSILON + 1;
double b_norm;
double start_time;
double finish_time;
double* A = malloc(sizeof(double) * N * N);
double* x = malloc(sizeof(double) * N);
double* b = malloc(sizeof(double) * N);
double* Axb = malloc(sizeof(double) * N);
generate_A(A, N);
generate_x(x, N);
generate_b(b, N);
b_norm = sqrt(calc_norm_square(b, N));
start_time = omp_get_wtime();
for (iter_count = 0; accuracy > EPSILON && iter_count < MAX_ITERATION_COUNT; ++iter_count)
{
calc_Axb(A, x, b, Axb, N);
calc_next_x(Axb, x, TAU, N);
accuracy = sqrt(calc_norm_square(Axb, N)) / b_norm;
}
finish_time = omp_get_wtime();
if (iter_count == MAX_ITERATION_COUNT)
printf("Too many iterations\n");
else
{
printf("Norm: %lf\n", sqrt(calc_norm_square(x, N)));
printf("Time: %lf sec\n", finish_time - start_time);
}
free(A);
free(x);
free(b);
free(Axb);
return 0;
}
void generate_A(double* A, int size)
{
#pragma omp parallel for
for (int i = 0; i < size; i++)
{
for (int j = 0; j < size; ++j)
A[i * size + j] = 1;
A[i * size + i] = 2;
}
}
void generate_x(double* x, int size)
{
#pragma omp parallel for
for (int i = 0; i < size; i++)
x[i] = 0;
}
void generate_b(double* b, int size)
{
#pragma omp parallel for
for (int i = 0; i < size; i++)
b[i] = N + 1;
}
double calc_norm_square(const double* vector, int size)
{
double norm_square = 0.0;
#pragma omp parallel for reduction(+ : norm_square)
for (int i = 0; i < size; ++i)
norm_square += vector[i] * vector[i];
return norm_square;
}
void calc_Axb(const double* A, const double* x, const double* b, double* Axb, int size) 
{
#pragma omp parallel for
for (int i = 0; i < size; ++i)
{
Axb[i] = -b[i];
for (int j = 0; j < N; ++j)
Axb[i] += A[i * N + j] * x[j];
}
}
void calc_next_x(const double* Axb, double* x, double tau, int size) 
{
#pragma omp parallel for
for (int i = 0; i < size; ++i)
x[i] -= tau * Axb[i];
}
