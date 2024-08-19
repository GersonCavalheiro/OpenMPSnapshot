#include<stdio.h>
#include<mpi.h>
#include<omp.h>
#include<stdlib.h>
int main(int argc,char **argv)
{
int i;
int rank,size,tid;
int *sync;
MPI_Request request;
MPI_Status status;
MPI_Init(&argc,&argv);
MPI_Comm_rank(MPI_COMM_WORLD,&rank);
MPI_Comm_size(MPI_COMM_WORLD,&size);
sync=(int *)calloc(size+1,sizeof(int));
sync[0]=10;
omp_set_num_threads(size);
MPI_Comm_rank(MPI_COMM_WORLD,&rank);
#pragma omp parallel private(tid)
{	
tid=omp_get_thread_num();
MPI_Bcast(&sync[tid],1,MPI_INT,tid,MPI_COMM_WORLD);
}
printf("rank %d c'est fini\n",rank);fflush(stdout);
free(sync);
MPI_Finalize();
}
