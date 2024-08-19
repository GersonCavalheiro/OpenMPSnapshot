#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>
int testConflict(int **board, int row, int col, int n) 
{
for (int i = 0; i < row; i++) 
{
if (board[i][col])
return 0;
}
for (int i = row, j = col; i >= 0 && j >= 0; i--, j--) 
{
if (board[i][j])
return 0;
}
for (int i = row, j = col; i >= 0 && j < n; i--, j++) 
{
if (board[i][j])
return 0;
}
return 1;
}
int solve(int **board, int row, int n, int rank) 
{
int solution = 0;
if (row == n)
return 1;
for (int col = 0; col < n; col++) 
{
if (testConflict(board, row, col, n)) 
{
board[row][col] = 1;
solution += solve(board, row + 1, n, rank);
board[row][col] = 0;
}
}
return solution;
}
int nrainhas(int n, int rank)
{
int solutions = 0;
#pragma omp parallel reduction(+:solutions)
{
int **board = (int **) malloc(n * sizeof(int *));
for (int i = 0; i < n; i++) 
{
board[i] = (int *) malloc(n * sizeof(int));
for (int j = 0; j < n; j++) 
{
board[i][j] = 0;       
}
}
board[0][rank] = 1;
#pragma omp for
for (int col = 0; col < n; col++) 
{
if (testConflict(board, 1, col, n))
{
board[1][col] = 1;
solutions += solve(board, 2, n, rank);   
board[1][col] = 0;
}
}
for (int i = 0; i < n; i++) 
free(board[i]);
free(board);
}
return solutions;
}
int main(int argc, char *argv[]) 
{
int n;
int solutions;
int myrank;
int aux;
int sum = 0;
int sendSolutions = 0;
int worldSize;
MPI_Status st;
MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
n = atoi(argv[1]);
if (argc < 3)
omp_set_num_threads(0);
else
omp_set_num_threads(atoi(argv[2]));
double start = omp_get_wtime();
for (int i = 0; i < n; i++)
{
if((i % worldSize) == myrank)
sendSolutions += nrainhas(n, i);
}
if (myrank != 0)
MPI_Send(&sendSolutions, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
MPI_Barrier(MPI_COMM_WORLD);
if(myrank == 0) 
{
for (int i = 1; i < worldSize; i++) 
{
MPI_Recv(&aux, 1, MPI_INT, i, MPI_ANY_TAG, MPI_COMM_WORLD, &st);
sum += aux;
}
printf("Numero de solucoes: %d\n", sum + sendSolutions);
double end = omp_get_wtime();
printf("Tempo de execucao: %fs\n", end - start);
}
MPI_Finalize();
return 0;  
}