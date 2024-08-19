#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "timer.h"
int     thread_count;
int     m, n;
double* A;
double* x;
double* y;
double elapsed = 0.0;
void Usage(char* prog_name);
void Gen_matrix(double A[], int m, int n);
void Read_matrix(char* prompt, double A[], int m, int n);
void Gen_vector(double x[], int n);
void Read_vector(char* prompt, double x[], int n);
void Print_matrix(char* title, double A[], int m, int n);
void Print_vector(char* title, double y[], double m, int padding);
void Omp_mat_vect ();
int main(int argc, char* argv[]) 
{
long       thread;
if (argc != 4) Usage(argv[0]);
thread_count = strtol(argv[1], NULL, 10);
m = strtol(argv[2], NULL, 10);
n = strtol(argv[3], NULL, 10);
#  ifdef DEBUG
printf("thread_count =  %d, m = %d, n = %d\n", thread_count, m, n);
#  endif
A = malloc(m*n*sizeof(double));
x = malloc(n*sizeof(double));
int SIZE_DOUBLE = 8;
if (m < SIZE_DOUBLE*thread_count) 
{
y = malloc(SIZE_DOUBLE*thread_count* sizeof(double));
} 
else 
{
y = malloc(m*sizeof(double));
}
Gen_matrix(A, m, n);
#  ifdef DEBUG
Print_matrix("We generated", A, m, n); 
#  endif
Gen_vector(x, n);
#  ifdef DEBUG
Print_vector("We generated", x, n, 1); 
#  endif
#pragma omp parallel num_threads(thread_count)
Omp_mat_vect();
#  ifdef DEBUG
if (m < SIZE_DOUBLE*thread_count) 
Print_vector("The product is", y, SIZE_DOUBLE*thread_count, SIZE_DOUBLE);
else 
Print_vector("The product is", y, m, 1);
#  endif
printf("%.10lf\n",elapsed);
free(A);
free(x);
free(y);
return 0;
}  
void Usage (char* prog_name) {
fprintf(stderr, "Usage: %s <thread_count> <m> <n>\n", prog_name);
exit(0);
}  
void Read_matrix(char* prompt, double A[], int m, int n) {
int             i, j;
printf("%s\n", prompt);
for (i = 0; i < m; i++) 
for (j = 0; j < n; j++)
scanf("%lf", &A[i*n+j]);
}  
void Gen_matrix(double A[], int m, int n) {
int i, j;
for (i = 0; i < m; i++)
for (j = 0; j < n; j++)
A[i*n+j] = random()/((double) RAND_MAX);
}  
void Gen_vector(double x[], int n) {
int i;
for (i = 0; i < n; i++)
x[i] = random()/((double) RAND_MAX);
}  
void Read_vector(char* prompt, double x[], int n) {
int   i;
printf("%s\n", prompt);
for (i = 0; i < n; i++) 
scanf("%lf", &x[i]);
}  
void defineVariaveisThread(int my_rank, int *local_m, int * my_first_row, int * my_last_row) 
{
double SIZE_DOUBLE = 8;
*local_m = m/thread_count;
if (m < SIZE_DOUBLE*thread_count) 
{
*my_first_row = my_rank*SIZE_DOUBLE;
*my_last_row = SIZE_DOUBLE*my_rank + *local_m;
} 
else 
{
*my_first_row = my_rank*(*local_m);
*my_last_row = *my_first_row + *local_m;
}
}
void Omp_mat_vect () 
{
long my_rank = omp_get_thread_num();
int i;
int j; 
int local_m, my_first_row, my_last_row;
defineVariaveisThread(my_rank, &local_m, &my_first_row, &my_last_row);
double start, finish;
double temp;
register int sub = my_rank*local_m*n;
#  ifdef DEBUG
printf("Thread %ld > local_m = %d, sub = %d\n",my_rank, local_m, sub);
printf("Thread %ld > my_first_row = %d, my_last_row = %d\n",my_rank,my_first_row,my_last_row);
#  endif
GET_TIME(start);
for (i = my_first_row; i < my_last_row; i++) 
{
y[i] = 0.0;
for (j = 0; j < n; j++) 
{
temp = A[sub++];
temp *= x[j];
y[i] += temp;
}
}
GET_TIME(finish);
#pragma omp critical
if ((finish-start) > elapsed)
elapsed = (finish-start);
}  
void Print_matrix( char* title, double A[], int m, int n) {
int   i, j;
printf("%s\n", title);
for (i = 0; i < m; i++) {
for (j = 0; j < n; j++)
printf("%6.3f ", A[i*n + j]);
printf("\n");
}
}  
void Print_vector(char* title, double y[], double m, int padding) 
{
int   i;
printf("%s\n", title);
for (i = 0; i < m; i += padding)
printf("%6.3f ", y[i]);
printf("\n");
}  
