#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N     1000000
#define PI    3.1415926535
#define DELTA .01415926535
int main (int argc, char *argv[])
{
int nthreads, tid, i;
float a[N], b[N];
omp_lock_t locka, lockb;
omp_init_lock(&locka);
omp_init_lock(&lockb);
for (i = 0; i < N; i++) {
a[i]=0;
b[i]=0;
}
#pragma omp parallel shared(a, b, nthreads, locka, lockb) private(tid, i)
{
tid = omp_get_thread_num();
#pragma omp master
{
nthreads = omp_get_num_threads();
printf("Number of threads = %d\n", nthreads);
}
printf("Thread %d starting...\n", tid);
#pragma omp barrier
#pragma omp sections nowait
{
#pragma omp section
{
printf("Thread %d updating a[]\n", tid);
for (i = 0; i < N; i++)
#pragma omp atomic
a[i] += DELTA * i;
printf("Thread %d updating b[]\n", tid);
for (i = 0; i < N; i++)
#pragma omp atomic
b[i] += DELTA + i;
}
#pragma omp section
{
printf("Thread %d updating b[]\n", tid);
for (i = 0; i < N; i++)
#pragma omp atomic
b[i] += PI * i;
printf("Thread %d adding b[] to a[]\n", tid);
for (i = 0; i < N; i++)
#pragma omp atomic
a[i] += PI + i;
}
}
}
printf("Sample results: %f %f %f %f\n", a[0], b[0], a[999999], b[999999]);
}