#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define N 100000
#define NUM_THREADS 2
int main(int argc, char *argv[])
{
double start,stop;
int x[N], y[N];
int i, j, my_num, my_place,tid;
start = omp_get_wtime();
for (i=0; i<N; i++) {
x[i] = N - i;
}
omp_set_num_threads(NUM_THREADS);
#pragma omp parallel private(j,i,tid,my_num,my_place)
{
tid=omp_get_thread_num();
for (j=tid*N/NUM_THREADS; j<tid*N/NUM_THREADS+N/NUM_THREADS; j++) {
my_num = x[j];
my_place = 0;
for (i=0; i<N; i++) {
if ( my_num > x[i] ) {
my_place++;
}            
}
y[my_place] = my_num;
}
stop = omp_get_wtime();
}
for (i=0; i<N; i++)
printf("%d\n", y[i]);
printf("time %f\n",stop-start);
return 0;
}
