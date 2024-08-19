#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
void Read_matrix(int local_mat[], int n, int my_rank, int p, MPI_Comm comm);
void Print_matrix(int local_mat[], int n, int my_rank, int p, MPI_Comm comm);
void Floyd(int local_mat[], int n, int my_rank, int p, MPI_Comm comm);
int Owner(int k, int p, int n);
void Copy_row(int local_mat[], int n, int p, int row_k[], int k);
void Print_row(int local_mat[], int n, int my_rank, int i);
int main(int argc, char* argv[]) {
int  n;
int* local_mat;
MPI_Comm comm;
int p, my_rank;
double gflops,numFlops;
struct timeval start_time, stop_time, elapsed_time;  
MPI_Init(&argc, &argv);
comm = MPI_COMM_WORLD;
MPI_Comm_size(comm, &p);
MPI_Comm_rank(comm, &my_rank);
if (my_rank == 0) {
printf("How many vertices?\n");
scanf("%d", &n);
}
MPI_Bcast(&n, 1, MPI_INT, 0, comm);
local_mat = malloc(n*n/p*sizeof(int));
Read_matrix(local_mat, n, my_rank, p, comm);
if (my_rank == 0) printf("\n");
MPI_Barrier(MPI_COMM_WORLD);
double  start = MPI_Wtime();
#pragma omp parallel
{
Floyd(local_mat, n, my_rank, p, comm);
}
MPI_Barrier(MPI_COMM_WORLD);
double stop = MPI_Wtime();
if (my_rank == 0){
printf("Completed in %f seconds\n" ,stop-start);
numFlops = 2.0f*n*n*n/1000000000.0f;
gflops = numFlops/(stop-start);
printf("GFlops :  %f .\n",gflops);  
}
free(local_mat);
MPI_Finalize();
return 0;
}  
void Read_matrix(int local_mat[], int n, int my_rank, int p, MPI_Comm comm) { 
int i, j;
int* temp_mat = NULL;
if (my_rank == 0) {
temp_mat = malloc(n*n*sizeof(int));
for (i = 0; i < n; i++) {
for (j = 0; j < n; j++)
if (i == j)
temp_mat[i*n+j]=0;
else {
if ((i==j+1)|| (j==i+1)||((i==0)&&(j==n-1))||((i==n-1)&&(j==0)))
temp_mat[i*n+j]=1;
else
temp_mat[i*n+j]=n; 
}
}
MPI_Scatter(temp_mat, n*n/p, MPI_INT, 
local_mat, n*n/p, MPI_INT, 0, comm);
free(temp_mat);
} else {
MPI_Scatter(temp_mat, n*n/p, MPI_INT, 
local_mat, n*n/p, MPI_INT, 0, comm);
}
}  
void Print_matrix(int local_mat[], int n, int my_rank, int p, MPI_Comm comm) {
int i, j;
int* temp_mat = NULL;
if (my_rank == 0) {
temp_mat = malloc(n*n*sizeof(int));
MPI_Gather(local_mat, n*n/p, MPI_INT, 
temp_mat, n*n/p, MPI_INT, 0, comm);
for (i = 0; i < n; i++) {
for (j = 0; j < n; j++)
printf("%d ", temp_mat[i*n+j]);
printf("\n");
}
free(temp_mat);
} else {
MPI_Gather(local_mat, n*n/p, MPI_INT, 
temp_mat, n*n/p, MPI_INT, 0, comm);
}
}  
void Floyd(int local_mat[], int n, int my_rank, int p, MPI_Comm comm) {
int global_k, local_i, global_j, temp;
int root;
int* row_k = malloc(n*sizeof(int));
for (global_k = 0; global_k < n; global_k++) {
root = Owner(global_k, p, n);
if (my_rank == root)
Copy_row(local_mat, n, p, row_k, global_k);
MPI_Bcast(row_k, n, MPI_INT, root, MPI_COMM_WORLD);
#pragma omp parallel for private(global_j,temp) shared(local_mat,global_k)
for (local_i = 0; local_i < n/p; local_i++)
for (global_j = 0; global_j < n; global_j++) {
temp = local_mat[local_i*n + global_k] + row_k[global_j];
if (temp < local_mat[local_i*n+global_j])
local_mat[local_i*n + global_j] = temp;
}
}
free(row_k);
}  
int Owner(int k, int p, int n) {
return (((p)*((k)+1)-1))/(n);
}  
void Copy_row(int local_mat[], int n, int p, int row_k[], int k) {
int j;
int local_k = k % (n/p);
for (j = 0; j < n; j++)
row_k[j] = local_mat[local_k*n + j];
}  
