#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
void Get_args(int argc, char* argv[], int* thread_count_p, 
int* m_p, int* n_p);
void Usage(char* prog_name);
void Gen_matrix(double A[], int m, int n);
void Read_matrix(char* prompt, double A[], int m, int n);
void Gen_vector(double x[], int n);
void Read_vector(char* prompt, double x[], int n);
void Print_matrix(char* title, double A[], int m, int n);
void Print_vector(char* title, double y[], double m);
void Omp_mat_vect(double A[], double x[], double y[],
int m, int n, int thread_count);
int main(int argc, char* argv[]) {
int     thread_count;
int     m, n;
double* A;
double* x;
double* y;
Get_args(argc, argv, &thread_count, &m, &n);
A = malloc(m*n*sizeof(double));
x = malloc(n*sizeof(double));
y = malloc(m*sizeof(double));
# ifdef DEBUG
Read_matrix("Enter the matrix", A, m, n);
Print_matrix("We read", A, m, n);
Read_vector("Enter the vector", x, n);
Print_vector("We read", x, n);
# else
Gen_matrix(A, m, n);
Gen_vector(x, n);
# endif
Omp_mat_vect(A, x, y, m, n, thread_count);
#  ifdef DEBUG
Print_vector("The product is", y, m);
#  else
#  endif
free(A);
free(x);
free(y);
return 0;
}  
void Get_args(int argc, char* argv[], int* thread_count_p, 
int* m_p, int* n_p)  {
if (argc != 4) Usage(argv[0]);
*thread_count_p = strtol(argv[1], NULL, 10);
*m_p = strtol(argv[2], NULL, 10);
*n_p = strtol(argv[3], NULL, 10);
if (*thread_count_p <= 0 || *m_p <= 0 || *n_p <= 0) Usage(argv[0]);
}  
void Usage (char* prog_name) {
fprintf(stderr, "usage: %s <thread_count> <m> <n>\n", prog_name);
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
void Omp_mat_vect(double A[], double x[], double y[],
int m, int n, int thread_count) {
int i, j;
double start, finish, elapsed;
start = omp_get_wtime();
#pragma omp parallel for num_threads(thread_count)  default(none) private(i, j)  shared(A, x, y, m, n)
for (i = 0; i < m; i++) {
y[i] = 0.0;
for (j = 0; j < n; j++)
y[i] += A[i*n+j]*x[j];
}
finish = omp_get_wtime();
elapsed = finish - start;
printf("Elapsed time = %e seconds\n", elapsed);
}  
void Print_matrix( char* title, double A[], int m, int n) {
int   i, j;
printf("%s\n", title);
for (i = 0; i < m; i++) {
for (j = 0; j < n; j++)
printf("%4.1f ", A[i*n + j]);
printf("\n");
}
}  
void Print_vector(char* title, double y[], double m) {
int   i;
printf("%s\n", title);
for (i = 0; i < m; i++)
printf("%4.1f ", y[i]);
printf("\n");
}  
