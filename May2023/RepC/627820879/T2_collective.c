#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <ctype.h>
void init(double *input, int length);
void print_matrix(double *matrix, int size);
void print_matrix_rows(double *matrix, int size, int rows);
int invoke_cuda_matrix_multiplication(double *h_a, double *h_b, double *h_c, int size, double **d_a, double **d_b, double **d_c);
float get_kernel_results(double *h_c, double **d_a, double **d_b, double **d_c, int size);
int main(int argc, char *argv[])
{
unsigned int N = 0;
if (argc == 2 && isdigit(argv[1][0]))
{
N = atoi(argv[1]);
}
else
{
printf("USAGE\n   %s [SIZE] \n", argv[0]);
return 0;
}
int world_size, id;
MPI_Init(NULL, NULL);
MPI_Comm_size(MPI_COMM_WORLD, &world_size);
MPI_Comm_rank(MPI_COMM_WORLD, &id);
#if defined DEBUG
printf("MPI world is received: size: %d, id %d \n", world_size, id);
#endif
double *h_a, *h_b, *h_c, *h_c2;
double *d_a, *d_b, *d_c;
int data_length = N * N * sizeof(double);
int data_size = N * N;
int rows_per_rank = N / world_size;
int partition_size = rows_per_rank * N;
double t1, t2;
double *local_a = (double *)malloc(partition_size * sizeof(double));
double *local_b = (double *)malloc(partition_size * sizeof(double));
double *local_c2 = (double *)malloc(partition_size * sizeof(double));
if (id == 0)
{
h_a = (double *)malloc(data_length);
h_b = (double *)malloc(data_length);
h_c = (double *)malloc(data_length);
h_c2 = (double *)malloc(data_length);
init(h_a, data_size);
init(h_b, data_size);
#if defined DEBUG
printf("Matrices were initialized\n");
#endif
t1 = MPI_Wtime();
invoke_cuda_matrix_multiplication(h_a, h_b, h_c, N, &d_a, &d_b, &d_c);
}
MPI_Scatter(h_a, partition_size, MPI_DOUBLE, local_a, partition_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
MPI_Scatter(h_b, partition_size, MPI_DOUBLE, local_b, partition_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#if defined DEBUG
printf("Data were distributed between ranks\n");
#endif
int i, j;
#pragma omp parallel for shared(local_a, local_b, local_c2, N, rows_per_rank) private(i, j)
for (i = 0; i < rows_per_rank; i++)
{
#pragma omp simd
for (j = 0; j < N; j++)
{
local_c2[i * N + j] = local_a[i * N + j] + local_b[i * N + j];
}
}
#if defined DEBUG
printf("%d: Rank have performed additions on own partition\n", id);
#endif
MPI_Gather(local_c2, partition_size, MPI_DOUBLE, h_c2, partition_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#if defined DEBUG
printf("%d: Matrix C was collected from ranks\n", id);
#endif
free(local_a);
free(local_b);
free(local_c2);
MPI_Barrier(MPI_COMM_WORLD);
if (id == 0)
{
printf("MATRIX SIZE: %i \n", N);
float elapsedTime = get_kernel_results(h_c, &d_a, &d_b, &d_c, N);
printf("Elapsed time for GPU: %f ms\n", elapsedTime);
double t2 = (MPI_Wtime() - t1) * 1000.0;
printf("Total elapsed time for MPI + GPU: %f ms\n", t2);
if (N <= 10)
{
printf("A+B on MPI result: \n");
print_matrix(h_c2, N);
printf("A*B on GPU result: \n");
print_matrix(h_c, N);
}
free(h_a);
free(h_b);
free(h_c);
free(h_c2);
}
MPI_Finalize();
#if defined DEBUG
printf("%d: MPI was shutdown\n", id);
#endif
return 0;
}
void init(double *input, int size)
{
int i;
for (i = 0; i < size; i++)
{
input[i] = rand() % 5;
}
}
void print_matrix(double *matrix, int size)
{
printf("Matrix items: \n");
int i, j;
for (i = 0; i < size; i++)
{
for (j = 0; j < size; j++)
printf("%f,", matrix[i * size + j]);
printf("\n");
}
}
void print_matrix_rows(double *matrix, int size, int rows)
{
printf("Matrix items: \n");
int i, j;
for (i = 0; i < rows; i++)
{
for (j = 0; j < size; j++)
printf("%f,", matrix[i * size + j]);
printf("\n");
}
}
