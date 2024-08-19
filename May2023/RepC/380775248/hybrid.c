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

#pragma omp parallel private(tid)
{
tid = omp_get_thread_num();
printf("Process: %d\t Thread: %d says hello!\n", rank, tid);
}

MPI_Finalize();

}