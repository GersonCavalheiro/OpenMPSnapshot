#include<stdlib.h>
#include<stdio.h>
#include<omp.h>
#define LARGE_NUMBER 100000
int item[LARGE_NUMBER];
int main() {
int j;
double start_time, run_time_parallel2, run_time_parallel1;
srand(5);
for (j=0; j<LARGE_NUMBER; j++){
item[j] = 1+(rand()%1000);
}
start_time = omp_get_wtime();
#pragma omp parallel
{
#pragma omp single
{
int i;
for (i=0; i<LARGE_NUMBER; i++){
#pragma omp task 
printf("%d ",item[i]);
}
printf("\n");
}
}
run_time_parallel1 = omp_get_wtime() - start_time;
start_time = omp_get_wtime();
#pragma omp parallel
{
#pragma omp single
{
int k;
#pragma omp task untied
{
for (k=0; k<LARGE_NUMBER; k++){
#pragma omp task
printf("%d ",item[k]);
}
}
}
}
run_time_parallel2 = omp_get_wtime() - start_time;
printf("\n");
printf("The time taken is : %f\n", run_time_parallel1);
printf("The time taken for untied is : %f\n", run_time_parallel2);
printf("\n");
return 0;
}
