#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#define SIZE_1 10000000
#define SIZE_2 20000000
#define SIZE_3 30000000
float y_1[SIZE_1], y_2[SIZE_2], y_3[SIZE_3];
float x_1[SIZE_1], x_2[SIZE_2], x_3[SIZE_3];
float a;
int main ()
{
double start_time, run_time;
srand(time(NULL));
omp_set_num_threads(omp_get_num_procs());
a = rand();
for (int i = 0; i < SIZE_3; ++i){
if(i < SIZE_1){
x_1[i] = rand();
y_1[i] = rand();
}
if (i < SIZE_2){
x_2[i] = rand();
y_2[i] = rand();
}
x_3[i] = rand();
y_3[i] = rand();
}
start_time = omp_get_wtime();
#pragma omp parallel  
{
#pragma omp for
for (int i = 0; i < SIZE_1; ++i)
y_1[i] = a*x_1[i] + y_1[i];
}
run_time = omp_get_wtime() - start_time;
printf("%d\t%f\n", SIZE_1, run_time);
start_time = omp_get_wtime();
#pragma omp parallel  
{
#pragma omp for
for (int i = 0; i < SIZE_2; ++i)
y_2[i] = a*x_2[i] + y_2[i];
}
run_time = omp_get_wtime() - start_time;
printf("%d\t%f\n", SIZE_2, run_time);
#pragma omp parallel  
{
#pragma omp for
for (int i = 0; i < SIZE_3; ++i)
y_3[i] = a*x_3[i] + y_3[i];
}
run_time = omp_get_wtime() - start_time;
printf("%d\t%f\n", SIZE_3, run_time);
return 0;
}