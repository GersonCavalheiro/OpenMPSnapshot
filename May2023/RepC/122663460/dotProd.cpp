#ifdef MPI
#include <mpi.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif 
#include <stdio.h>
#include <stdlib.h>
#define VECLEN 100
#define NUMTHREADS 8
int main (int argc, char* argv[])
{
int i, myid, tid, numprocs, len=VECLEN, threads=NUMTHREADS;
double *a, *b;
double mysum, allsum, sum, psum;
MPI_Init (&argc, &argv);
MPI_Comm_size (MPI_COMM_WORLD, &numprocs);
MPI_Comm_rank (MPI_COMM_WORLD, &myid);
if (myid == 0)
printf("Starting omp_dotprod_hybrid. Using %d tasks...\n",numprocs);
a = (double*) malloc (len*threads*sizeof(double));
b = (double*) malloc (len*threads*sizeof(double));
for (i=0; i<len*threads; i++) {
a[i]=1.0;
b[i]=a[i];
}
sum = 0.0;
#pragma omp parallel private(i,tid,psum) num_threads(threads)
{
psum = 0.0;
tid = omp_get_thread_num();
if (tid ==0)
{
threads = omp_get_num_threads();
printf("Task %d using %d threads\n",myid, threads);
}
#pragma omp for reduction(+:sum)
for (i=0; i<len*threads; i++)
{
sum += (a[i] * b[i]);
psum = sum;
}
printf("Task %d thread %d partial sum = %f\n",myid, tid, psum);
}
mysum = sum;
printf("Task %d partial sum = %f\n",myid, mysum);
MPI_Reduce (&mysum, &allsum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
if (myid == 0) 
printf ("Done. Hybrid version: global sum  =  %f \n", allsum);
free (a);
free (b);
MPI_Finalize();
} 
