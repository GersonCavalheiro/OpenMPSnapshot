#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N       50
#define CHUNKSIZE   5
int main (int argc, char *argv[]) 
{
int i, chunk, tid;
float a[N], b[N], c[N];
char first_time;
for (i=0; i < N; i++)
a[i] = b[i] = i * 1.0;
chunk = CHUNKSIZE;
first_time = 'y';
#pragma omp parallel for     shared(a,b,c,chunk)        private(i,tid)             schedule(static,chunk)     firstprivate(first_time)
for (i=0; i < N; i++)
{
if (first_time == 'y')
{
tid = omp_get_thread_num();
first_time = 'n';
}
c[i] = a[i] + b[i];
printf("tid= %d i= %d c[i]= %f\n", tid, i, c[i]);
}
}
