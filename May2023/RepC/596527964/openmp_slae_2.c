#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N 5000
#define EPSILON 1E-7
#define TAU 1E-5
#define MAX_ITERATION_COUNT 1000000
void set_matrix_part(int* line_counts, int* line_offsets, int size, int thread_count);
void generate_A(double* A, int size);
void generate_x(double* x, int size);
void generate_b(double* b, int size);
double calc_norm_square(const double* vector, int size);
void calc_Axb(const double* A, const double* x, const double* b, double* Axb, int size);
void calc_next_x(const double* Axb, double* x, double tau, int size);
int main(int argc, char **argv)
{
int iter_count = 0;
int thread_id;
int thread_count = omp_get_max_threads();
double accuracy = EPSILON + 1;
double b_norm;
double start_time;
double finish_time;
int* line_counts = malloc(sizeof(int) * thread_count);
int* line_offsets = malloc(sizeof(int) * thread_count); 
double* A = malloc(sizeof(double) * N * N);
double* x = malloc(sizeof(double) * N);
double* b = malloc(sizeof(double) * N);
double* Axb = malloc(sizeof(double) * N);
set_matrix_part(line_counts, line_offsets, N, thread_count);
generate_A(A, N);
generate_x(x, N);
generate_b(b, N);
b_norm = sqrt(calc_norm_square(b, N));
start_time = omp_get_wtime();
#pragma omp parallel private(thread_id)
{
thread_id = omp_get_thread_num();
for (iter_count = 0; accuracy > EPSILON && iter_count < MAX_ITERATION_COUNT; ++iter_count)
{
calc_Axb(A + line_offsets[thread_id] * N, x, b + line_offsets[thread_id], 
Axb + line_offsets[thread_id], line_counts[thread_id]);
#pragma omp barrier
calc_next_x(Axb + line_offsets[thread_id], x + line_offsets[thread_id], 
TAU, line_counts[thread_id]);
#pragma omp single
accuracy = 0;
#pragma omp atomic
accuracy += calc_norm_square(Axb + line_offsets[thread_id], line_counts[thread_id]);
#pragma omp barrier
#pragma omp single
accuracy = sqrt(accuracy) / b_norm;
}
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
void set_matrix_part(int* line_counts, int* line_offsets, int size, int thread_count) 
{
int offset = 0;
for (int i = 0; i < thread_count; ++i)
{
line_counts[i] = size / thread_count;
if (i < size % thread_count)
++line_counts[i];
line_offsets[i] = offset;
offset += line_counts[i];
}
}
void generate_A(double* A, int size)
{
for (int i = 0; i < size; i++)
{
for (int j = 0; j < size; ++j)
A[i * size + j] = 1;
A[i * size + i] = 2;
}
}
void generate_x(double* x, int size)
{
for (int i = 0; i < size; i++)
x[i] = 0;
}
void generate_b(double* b, int size)
{
for (int i = 0; i < size; i++)
b[i] = N + 1;
}
double calc_norm_square(const double* vector, int size)
{
double norm_square = 0.0;
for (int i = 0; i < size; ++i)
norm_square += vector[i] * vector[i];
return norm_square;
}
void calc_Axb(const double* A, const double* x, const double* b, double* Axb, int size) 
{
for (int i = 0; i < size; ++i)
{
Axb[i] = -b[i];
for (int j = 0; j < N; ++j)
Axb[i] += A[i * N + j] * x[j];
}
}
void calc_next_x(const double* Axb, double* x, double tau, int size) 
{
for (int i = 0; i < size; ++i)
x[i] -= tau * Axb[i];
}
