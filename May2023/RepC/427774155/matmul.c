#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <omp.h>
#define N 512
#define M 512
#define K 512
int A[N][M], B[M][K], C[N][K];
void matmul_serial();
void matmul_omp_taskloop();
static
void init_arrays()
{
int i, j;
for (i = 0; i < N; i++)
for (j = 0; j < M; j++)
A[i][j] = 2;
for (i = 0; i < M; i++)
for (j = 0; j < K; j++)
B[i][j] = 2;
}
static
void reset_results()
{
int i, j;
for (i = 0; i < N; i++)
for (j = 0; j < K; j++)
C[i][j] = 0;
}
static
void print_results()
{
int i, j;
for (i = 0; i < N; i++)
{
for (j = 0; j < K; j++)
printf("%d ", C[i][j]);
printf("\n");
}
}
void matmul_serial()
{
int i, j, k;
for (i = 0; i < N; ++i) 
for (j = 0; j < K; ++j) 
for (k = 0; k < M; ++k) 
C[i][j] += A[i][k] * B[k][j];
}
void matmul_omp_taskloop()
{
int i, j, k;
#pragma omp parallel num_threads(4)
{
#pragma omp single
{
for (i = 0; i < N; ++i) 
{    
#pragma omp taskloop private(i,k) shared(A,B)
for (j = 0; j < K; ++j) 
{    
for (k = 0; k < M; ++k) 
{
C[i][j] += A[i][k] * B[k][j];
}
}
}
}
}
}
int main()
{
int i, j;
double exectime, exectimepar;
struct timeval start, end;
if (N < 1 || M < 1 || K < 1)
{
fprintf(stderr, "Wrong dimensions; exiting.\n");
exit(1);
}
printf("matmul (A[%d][%d], B[%d][%d], C[%d][%d])\n",
N, M, M, K, N, K);
init_arrays();
gettimeofday(&start, NULL);
matmul_serial();
gettimeofday(&end, NULL);
exectime = (double) (end.tv_usec - start.tv_usec) / 1000000 
+ (double) (end.tv_sec - start.tv_sec); 
reset_results(); 
gettimeofday(&start, NULL);
matmul_omp_taskloop();
gettimeofday(&end, NULL);
exectimepar = (double) (end.tv_usec - start.tv_usec) / 1000000 
+ (double) (end.tv_sec - start.tv_sec); 
printf("Execution time (serial): %lf\n", exectime);
printf("Execution time (parallel): %lf\n", exectimepar);
}
