#include<stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#include<omp.h>

#define MAX_THREADS 1000
#define MAX(a, b) (a>b)?a:b

unsigned long int n;

void daxpy_uniprocess(double* X, double* Y, int a)
{
unsigned long int i;
for(i=0; i<n; i++)
X[i] = a*X[i]+Y[i];
}

void daxpy_multi_process(double* X, double* Y, int a, unsigned long int num_threads)
{
omp_set_num_threads(num_threads);
int thread_id;
unsigned long int i, index;

#pragma omp parallel private(i)
{
thread_id = omp_get_thread_num();
for(i=0; i<n/num_threads; i++)
{
index = i*num_threads+thread_id;
X[index] = a*X[index]+Y[index];
}
}
}

int main()
{
n = 1<<16;
double X[n], Y[n];
unsigned long int i;

for(i=0; i<n; i++)
{
X[i] = rand() + rand() / RAND_MAX;
Y[i] = rand() + rand() / RAND_MAX;
}

int a = 314;
int min_val_thread;

double start, end;
double min_time;

printf("Time taken by uniprocessor: ");
start = omp_get_wtime();
daxpy_uniprocess(X, Y, a);
end = omp_get_wtime();
printf("Start time: %lfs  End time: %lfs  Time elapsed: %lfs\n\n", start, end, end - start);
min_time = end - start;
double uniProcTime = min_time;
min_val_thread = 1;


for(i=2; i<=MAX_THREADS; i++)
{
printf("Time taken by %d threads: ", i);
start = omp_get_wtime();
daxpy_uniprocess(X, Y, a);
end = omp_get_wtime();
printf("Start time: %lf  End time: %lf  Time elapsed: %lfs\n", start, end, end - start);
if(end - start < min_time)
{
min_time = end - start;
min_val_thread = i;
}
}
printf("\n");

printf("Minimum time taken: %lfs\nNumber of threads: %d\n\n", min_time, min_val_thread);
printf("Maximum Speedup: %lf\n", uniProcTime/min_time);

}
