#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <stdlib.h>
#define BILLION 1000000000L
int main(int argc, char *argv[])
{
int i;
int N = atoi(argv[1]);
double a[N], b[N], c[N], d[N],e[N];
struct timespec start,end;
double time_taken = 0;
for(i=0; i<N; i++)
{
a[i] = i*2.0;
b[i] = i + a[i]*2.5;
}
clock_gettime( CLOCK_MONOTONIC, &start);
#pragma omp parallel num_threads(1) shared(a,b,c,d,e) private(i)
{
#pragma omp sections nowait
{
#pragma omp section
for(i=0;i<N;i++)
c[i]=a[i]+b[i];
#pragma omp section
for(i=0;i<N;i++)
d[i]=a[i]*b[i];
#pragma omp section
for(i=0;i<N;i++)
e[i]=(a[i]*a[i])+(b[i]*b[i]); 
}
}
clock_gettime( CLOCK_MONOTONIC, &end);
time_taken = ( end.tv_sec - start.tv_sec )+ (double)(end.tv_nsec - start.tv_nsec ) / (double)BILLION;
printf("values of a[%d]=%.1f, b[%d]=%.1f, c[%d]=%.1f, d[%d]=%.1f, e[%d]=%.1f\nTime taken for execution: %f\n",N-1,a[N1],N-1,b[N-1],N-1,c[N-1],N-1,d[N-1],N-1,e[N-1],time_taken);
} 
