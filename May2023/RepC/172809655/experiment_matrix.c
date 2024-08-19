#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "vector.h"
#include "matrix.h"
int main(int argc, char *argv[]){
if(argc < 3){
printf("WARNING: Missing arguments. \n");
printf("Usage: test_linalg N n_trials \n");
printf("    N: Matrices are NxN and vectors are Nx1\n");
printf("    n_trials: Number of experiment trials to run\n");
return 1;
}
int N = atoi(argv[1]);
int n_trials = atoi(argv[2]);
int n_threads;
#pragma omp parallel
{
n_threads = omp_get_num_threads();
} 
printf("matrix, %4d, %4d, %4d", n_threads, n_trials, N);
Matrix A;
Vector x;
Vector b;
allocate_Matrix(&A, N, N);
allocate_Vector(&x, N);
allocate_Vector(&b, N);
rand_fill_Matrix(&A);
rand_fill_Vector(&x);
zero_fill_Vector(&b);
double t_start, t_end;
double t_total, t_avg;
t_total = 0.0;
t_avg = 0.0;
for(int i=0; i<n_trials; i++){
t_start = omp_get_wtime();
matvec(&A, &x, &b);
t_end = omp_get_wtime();
t_total += t_end - t_start;
}
t_avg = t_total / n_trials;
printf(", %.10f", t_avg);
t_total = 0.0;
t_avg = 0.0;
for(int i=0; i<n_trials; i++){
t_start = omp_get_wtime();
matvec_triangular(&A, &x, &b);
t_end = omp_get_wtime();
t_total += t_end - t_start;
}
t_avg = t_total / n_trials;
printf(", %.10f", t_avg);
t_total = 0.0;
t_avg = 0.0;
for(int i=0; i<n_trials; i++){
t_start = omp_get_wtime();
matvec_triangular_guided(&A, &x, &b);
t_end = omp_get_wtime();
t_total += t_end - t_start;
}
t_avg = t_total / n_trials;
printf(", %.10f", t_avg);
printf("\n");
deallocate_Matrix(&A);
deallocate_Vector(&x);
deallocate_Vector(&b);
return 0;
}
