#include <omp.h>
#include <stdio.h>
#define  N 10000 
void main()
{
double start_time,run_time; 
int i;
int a[N],b[N],c[N];
start_time=omp_get_wtime();
#pragma omp parallel num_threads(1)
{
int bstart, bend, blen, numth, tid, i;
numth=omp_get_num_threads();
tid=omp_get_thread_num();
blen=N/numth;
if(tid < N % numth) {
blen++;
bstart=blen*tid;
}
else {
bstart=blen*tid+N%numth;
}
bend=bstart+blen-1;
for(i=bstart; i<=bend; i++) 
{
c[i]=i;
b[i]=i;
}
for(i=bstart; i<=bend; i++) { 
a[i]=b[i]+c[i];
}
}
run_time=omp_get_wtime()-start_time;
printf("Execution Time=%lf\n", run_time);
printf("Value of Dynamic=%d\n", omp_get_dynamic());
}