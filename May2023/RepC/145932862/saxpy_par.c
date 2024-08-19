#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
int main () {
srand(time(NULL));
int size = 1000000;
int *y = (int *)malloc(sizeof(int)*size);
int *x = (int *)malloc(sizeof(int)*size);
int *z = (int *)malloc(sizeof(int)*size);
int a = 5;
double start_time, run_time;
for(int i = 0; i < size; i++) {
y[i] = rand() % 1000;
x[i] = rand() % 1000;
}
omp_set_num_threads(2);
start_time = omp_get_wtime();
#pragma omp parallel for
for(int i = 0; i < size; i++) {
z[i] = a*x[i] + y[i];
}
run_time = omp_get_wtime() - start_time;
printf("\n Tiempo con 1000000 %f  \n",run_time);
start_time = omp_get_wtime();
#pragma omp parallel for
for(int i = 0; i < 500000; i++) {
z[i] = a*x[i] + y[i];
}
run_time = omp_get_wtime() - start_time;
printf("\n Tiempo con 500000 %f  \n",run_time);
start_time = omp_get_wtime();
#pragma omp parallel for
for(int i = 0; i < 100000; i++) {
z[i] = a*x[i] + y[i];
}
run_time = omp_get_wtime() - start_time;
printf("\n Tiempo con 100000 %f  \n",run_time);
}
