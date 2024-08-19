#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define CHUNKSIZE 10
#define N 100
int main () {
int nthreads, tid, i, chunk;
int a[N], b[N], c[N];
for (i=0; i<N; i++)
a[i] = b[i] = i * 1;
chunk = CHUNKSIZE;
#pragma omp parallel shared(a, b, c, nthreads, chunk) private(i, tid)
{
tid = omp_get_thread_num();
if (tid==0) {
nthreads = omp_get_num_threads();
printf("Number of threads = %d\n", nthreads);
}
printf("Thread %d starting...\n", tid);
#pragma omp for schedule(dynamic, chunk)
for (i=0; i<N; i++) {
c[i] = a[i] + b[i];
printf("Thread %d: c[%d] = %d\n", tid, i, c[i]);
}
}
return 0;
}
