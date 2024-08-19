#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include "omp.h"

#define COMM MPI_COMM_WORLD
#define MASTER_P 0
#define MASTER_T 0

int main(void) {

int size, rank, tid, provided, dataToSend = 0, dataToRecv = 0;

MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided);

MPI_Comm_rank(COMM, &rank);
MPI_Comm_size(COMM, &size);

MPI_Status status;

#pragma omp parallel private(tid)
{
tid = omp_get_thread_num();
if (rank == MASTER_P) {
#pragma omp master
{   
dataToSend = 77;
MPI_Send(&dataToSend, 1, MPI_INT, 1, 555, COMM);
}
#pragma omp barrier
} else {
#pragma omp master
MPI_Recv(&dataToRecv, 1, MPI_INT, MASTER_P, 555, COMM, &status);
#pragma omp barrier
}
}

if (rank == 1)
printf("DataToRecv = %d\n", dataToRecv);

MPI_Finalize();

}