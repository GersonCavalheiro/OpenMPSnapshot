#include <omp.h>
#define N     50
main ()
{
int i, n, nthreads, tid;
float a[N], b[N], c[N];
for (i=0; i < N; i++)
a[i] = b[i] = i * 1.0;
n = N;
#pragma omp parallel shared(a,b,c,n) private(i,tid,nthreads)
{
tid = omp_get_thread_num();
printf("Thread %d starting...\n",tid);
#pragma omp sections nowait
{
#pragma omp section
for (i=0; i < n/2; i++)
{
c[i] = a[i] + b[i];
printf("tid= %d i= %d c[i]= %f\n",tid,i,c[i]);
}
#pragma omp section
for (i=n/2; i < n; i++)
{
c[i] = a[i] + b[i];
printf("tid= %d i= %d c[i]= %f\n",tid,i,c[i]);
}
}  
if (tid == 0)
{
nthreads = omp_get_num_threads();
printf("Number of threads = %d\n", nthreads);
}
}  
}
