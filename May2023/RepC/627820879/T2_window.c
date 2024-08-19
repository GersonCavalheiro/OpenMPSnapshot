#define N 4096
#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
void init(double * input,int length);
void print_matrix(double * matrix, int rows, int cols, int id);
int invoke_cuda_matrix_multiplication(double *a , double * b, double * c, int size, double **, double **, double**);
void get_results (double *, double**, double**, double**,int);
int main()
{
MPI_Init(NULL,NULL);
int id;
int world_size;
MPI_Comm_size(MPI_COMM_WORLD,&world_size);
MPI_Comm_rank(MPI_COMM_WORLD,&id);
#if defined(DEBUG)
printf("%d: DEBUG is defined\n", id);
#endif
double * h_a;
double * h_b;
double * h_c; 
int data_size = N * N;
int partition_size = N * N / world_size;
double * h_pa = (double *)malloc(partition_size * sizeof(double));
double * h_pb = (double *)malloc(partition_size * sizeof(double));
double * h_pc2 = (double *)malloc(partition_size * sizeof(double));
double ** d_a, ** d_b, ** d_c;
double t1, t2;
MPI_Win win_a, win_b, win_c2;
MPI_Win_create(h_pa, partition_size, MPI_DOUBLE, MPI_INFO_NULL, MPI_COMM_WORLD, &win_a);
MPI_Win_create(h_pb, partition_size, MPI_DOUBLE, MPI_INFO_NULL, MPI_COMM_WORLD, &win_b);
MPI_Win_create(h_pc2, partition_size, MPI_DOUBLE, MPI_INFO_NULL, MPI_COMM_WORLD, &win_c2);
MPI_Win_fence(0, win_a);
MPI_Win_fence(0, win_b);
MPI_Win_fence(0, win_c2);
if(id==0)   
{
h_a = (double *)malloc(data_size * sizeof(double));
h_b = (double *)malloc(data_size * sizeof(double));
h_c = (double *)malloc(data_size * sizeof(double)); 
d_a = (double **)malloc(sizeof(double *));
d_b = (double **)malloc(sizeof(double *));
d_c = (double **)malloc(sizeof(double *));
init(h_a, N*N);
init(h_b, N*N);
#if defined(DEBUG)
printf("%d: Initializes matrices A and B: \n", id);
printf("%d: Matrix A: \n", id);
print_matrix(h_a, N, N, id);
printf("%d: Matrix B: \n", id);
print_matrix(h_b, N, N, id);
#endif
t1 = MPI_Wtime();
invoke_cuda_matrix_multiplication(h_a, h_b, h_c, N, d_a, d_b, d_c);
h_pa = h_a;
h_pb = h_b;
MPI_Win_lock_all(0, win_a);
for (int i = 1; i < world_size; i++) {
MPI_Put(h_a + i * partition_size, partition_size, MPI_DOUBLE, i, 0, partition_size, MPI_DOUBLE, win_a);
}
MPI_Win_unlock_all(win_a);
MPI_Win_lock_all(0, win_b);
for (int i = 1; i < world_size; i++) {
MPI_Put(h_b + i * partition_size, partition_size, MPI_DOUBLE, i, 0, partition_size, MPI_DOUBLE, win_b);
}
MPI_Win_unlock_all(win_b);
}
MPI_Win_fence(0, win_a);
MPI_Win_fence(0, win_b);
#if defined(DEBUG)
printf("%d: Partitioned matrices of A and B: \n", id);
printf("%d: Matrix PA: \n", id);
print_matrix(h_pa, N / world_size, N, id);
printf("%d: Matrix PB: \n", id);
print_matrix(h_pb, N / world_size, N, id);
#endif
int i;
#pragma omp parallel for shared(h_pc2, h_pa, h_pb) private(i)
for (i = 0; i < partition_size; i++) {
h_pc2[i] = h_pa[i] + h_pb[i];
}
#if defined(DEBUG)
printf("%d: Partitioned Matrix PC2: \n", id);
print_matrix(h_pc2, N / world_size, N, id);
#endif
MPI_Win_fence(0, win_c2);
if(id==0)
{
double * h_c2 = (double *)malloc(data_size * sizeof(double));
memcpy(h_c2, h_pc2, partition_size * sizeof(double));
#if defined(DEBUG)
printf("%d: Initialized Matrix C2: \n", id);
print_matrix(h_c2, N, N, id);
#endif
MPI_Win_lock_all(0, win_c2);
for (int i = 1; i < world_size; i++) {
MPI_Get(h_c2 + i * partition_size, partition_size, MPI_DOUBLE, i, 0, partition_size, MPI_DOUBLE, win_c2);
}
MPI_Win_unlock_all(win_c2);
get_results(h_c, d_a, d_b, d_c, data_size * sizeof(double));
t2 = MPI_Wtime();
printf("%d: Time taken in seconds: %f\n", id, t2 - t1);
if (N < 10) {
printf("%d: Matrix C2: \n", id);
print_matrix(h_c2, N, N, id);
printf("%d: Matrix C: \n", id);
print_matrix(h_c, N, N, id);
}
free(h_a);
free(h_b);
free(h_c2);
free(h_c);
free(d_a);
free(d_b);
free(d_c);
}
MPI_Win_free(&win_a);
MPI_Win_free(&win_b);
MPI_Win_free(&win_c2);
MPI_Finalize();
}
void init(double * input, int size)
{
int i;
for(i=0;i<size;i++)
{
input[i]=rand()%5;
}
}
void print_matrix(double * matrix, int rows, int cols, int id)
{
printf("%d: Matrix items: \n", id);
int i,j;
for(i=0;i<rows;i++)
{
printf("%d: ", id);
for(j=0;j<cols;j++)
printf("%f,",matrix[i*cols+j]);
printf("\n");
}
}
