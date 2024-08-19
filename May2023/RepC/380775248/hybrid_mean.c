#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include "omp.h"


#define COMM MPI_COMM_WORLD
#define MASTER_P 0
#define MASTER_T 0
#define N 20000

int main(void) {

int size, rank, tid, provided, i, elementsPerThread;
double *array, *local_array;

MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);

MPI_Comm_rank(COMM, &rank);
MPI_Comm_size(COMM, &size);

if (rank == MASTER_P) {
array = (double *)malloc(N * sizeof(double));
for (int i = 0; i < N; i++)
array[i] = 2.3;
}

int chunk = N / size;
local_array = (double *)malloc(chunk* sizeof(double));
MPI_Scatter(array, chunk, MPI_DOUBLE, local_array, chunk, MPI_DOUBLE, MASTER_P, COMM);

double local_sum = 0.0;

#pragma omp parallel shared(local_sum) private(i)
{
elementsPerThread = chunk / omp_get_num_threads();
#pragma omp for schedule(static,elementsPerThread) reduction(+:local_sum)
for (i = 0; i < chunk; i++)
local_sum += local_array[i];
}

double avg = 0.0;
MPI_Reduce(&local_sum, &avg, 1, MPI_DOUBLE, MPI_SUM, MASTER_P, COMM);

if (rank == MASTER_P)
printf("Average is %.20f\n", avg/N);

MPI_Finalize();

}