#include <omp.h>	
#include <stdio.h>
#include <stdlib.h>
# include <time.h>
#define CHUNKSIZE   100
#define N       100000
int main (int argc, char *argv[]) 
{
int nthreads, tid, i, chunk;
float a[N], b[N], c[N];
for (i=0; i < N; i++)
a[i] = b[i] = i * 1.0;
chunk = CHUNKSIZE;
clock_t start_t, end_t, total_t;
start_t = clock(); 
double start = omp_get_wtime(); 
#pragma omp parallel shared(a,b,c,nthreads,chunk) private(i,tid)
{
tid = omp_get_thread_num();	
if (tid == 0)
{
nthreads = omp_get_num_threads();	
printf("Number of threads = %d\n", nthreads);
}
printf("Thread %d starting...\n",tid);
#pragma omp for schedule(dynamic,chunk)
for (i=0; i<N; i++)
{
c[i] = a[i] + b[i];
printf("Thread %d: c[%d]= %f\n",tid,i,c[i]);
}	
}  
double end = omp_get_wtime(); 
end_t = clock(); 
total_t = (double)(end_t - start_t)/ CLOCKS_PER_SEC;
printf("time using omp clock %g \n", end - start);
printf("\n Total time taken by openmp in  seconds: %f\n", (double)(total_t ) );
}
