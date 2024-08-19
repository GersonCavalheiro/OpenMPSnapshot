#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include "omp.h"

#define COMM MPI_COMM_WORLD

int main(void) {

int size, rank, tid, provided;

MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);

MPI_Comm_rank(COMM, &rank);
MPI_Comm_size(COMM, &size);

MPI_Status status;

int a, b = 0;

#pragma omp parallel private(tid, a, b)
{
tid = omp_get_thread_num();
a = 3 * tid + 1;

if (rank == 0)
MPI_Send(&a, 1, MPI_INT, 1, 777, COMM);
else
MPI_Recv(&b, 1, MPI_INT, 0, 777, COMM, &status);

if (rank == 1)
printf("thread = %d\tb = %d\n", tid, b);

}


MPI_Finalize();

}