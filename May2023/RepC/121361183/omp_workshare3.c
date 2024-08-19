#include <omp.h>
#define N       50
#define CHUNK   5
main ()  {
int i, n, chunk, tid;
float a[N], b[N], c[N];
for (i=0; i < N; i++)
a[i] = b[i] = i * 1.0;
n = N;
chunk = CHUNK;
#pragma omp parallel for     shared(a,b,c,n)            private(i,tid)             schedule(static,chunk)
{
tid = omp_get_thread_num();
for (i=0; i < n; i++)
{
c[i] = a[i] + b[i];
printf("tid= %d i= %d c[i]= %f\n", tid, i, c[i]);
}
}  
}
