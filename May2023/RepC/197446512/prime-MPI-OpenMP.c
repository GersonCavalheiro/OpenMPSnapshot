#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "myheader.h"
#include "mpi.h"
#include <omp.h>

int main(int argc, char** argv){

int n=100000;
int numprimes = 0;
int size,rank;
int i;

MPI_Init(&argc, &argv);
MPI_Comm_size(MPI_COMM_WORLD, &size);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);


int* split = (int*) malloc((size+1) * sizeof(int));

for(i=0;i<=size;i++){
if(i == size) split[i] = n;
else split[i] = (n/size) * i;
}	

int tid;
#pragma omp parallel default(shared) private(tid) num_threads(4)
{	
double start = omp_get_wtime();
tid = omp_get_thread_num();

if (tid==0 && rank==0){
int nthreads = omp_get_num_threads();
printf("Number of ranks: %d\n", size);
printf("Number of threads: %d\n", nthreads);
}
#pragma omp for reduction(+:numprimes)
for(i = split[rank]; i <= split[rank+1]; i++){
if (is_prime(i) == 1)
numprimes++;
}

double end = omp_get_wtime();
printf("Rank ID: %d Thread ID: %d Time: %f\n",rank,tid,end-start);
}
int* numprimesSend = (int*) malloc(sizeof(int));
numprimesSend[0]=numprimes;
int* finalNumprimes = (int*) malloc(sizeof(int));
MPI_Reduce(numprimesSend,finalNumprimes,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);

if(rank == 0)
printf("Number of Primes: %d\n",finalNumprimes[0]);

MPI_Finalize();

return 0;

}

int is_prime(int n)
{

if      (n == 0) return 0;
else if (n == 1) return 0;
else if (n == 2) return 1;

int i;
for(i=2;i<=(int)(sqrt((double) n));i++)
if (n%i==0) return 0;

return 1;
}

