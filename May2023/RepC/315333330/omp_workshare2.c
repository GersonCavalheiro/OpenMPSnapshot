#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N     50
int main (int argc, char *argv[]) {
int i, nthreads, tid;
float a[N], b[N], c[N];
for (i=0; i<N; i++)
a[i] = b[i] = i * 1.0;
#pragma omp parallel shared(a,b,nthreads) private(c,i,tid)
{
tid = omp_get_thread_num();
if (tid == 0)
{
nthreads = omp_get_num_threads();
printf("Number of threads = %d\n", nthreads);
}
printf("Thread %d starting...\n",tid);
#pragma omp sections nowait
{
#pragma omp section
{
printf("Thread %d doing section 1\n",tid);
for (i=0; i<N; i++)
{
c[i] = a[i] + b[i];
printf("Thread %d: c[%d]= %f\n",tid,i,c[i]);
}
}
#pragma omp section
{
printf("Thread %d doing section 2\n",tid);
for (i=0; i<N; i++)
{
c[i] = a[i] * b[i];
printf("Thread %d: c[%d]= %f\n",tid,i,c[i]);
}
}
}  
printf("Thread %d done.\n",tid); 
}  
return 0;
}
