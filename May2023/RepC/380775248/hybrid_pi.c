#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include "omp.h"

#define COMM MPI_COMM_WORLD
#define N 1000000
#define MASTER 0

double f(double x) {
return ( 4.0 / (1.0 + x * x) );
}

int main(void) {

int size, rank, tid, provided;

MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided);

MPI_Comm_rank(COMM, &rank);
MPI_Comm_size(COMM, &size);

int chunk = N / size;

double local_sum = 0.0;
#pragma omp parallel shared(rank)
{
#pragma omp for schedule(dynamic) reduction(+:local_sum)
for (int i = rank * chunk; i < (rank+1) * chunk; i++)
local_sum += f( (double)i / N );
}

double pi = 0.0;
MPI_Reduce(&local_sum, &pi, 1, MPI_DOUBLE, MPI_SUM, MASTER, COMM);

if (rank == MASTER)
printf("PI = %.20f\n", pi / N);

MPI_Finalize();

}