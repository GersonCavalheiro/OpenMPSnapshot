#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#define SIZE_M 10000000
float x[SIZE_M], y[SIZE_M];
float a = 0;
int main (){
double start_time, run_time;
srand(time(NULL));
int procs = omp_get_num_procs();
omp_set_num_threads(procs);
#pragma omp parallel  
{
#pragma omp for
for (int i = 0; i < SIZE_M; ++i)
{
x[i] = rand();
y[i] = rand();
}
}
start_time = omp_get_wtime();
#pragma omp parallel  
{
float element = 0;
#pragma omp for reduction(+:a) private(element)
for (int i = 0; i < SIZE_M; ++i)
{
element = x[i]*y[i];
a += element;
}
}
run_time = omp_get_wtime() - start_time;
printf("x*y = %f in %f seconds\n", a, run_time);
return 0;
}