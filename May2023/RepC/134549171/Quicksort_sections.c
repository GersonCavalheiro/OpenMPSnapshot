#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <sched.h>
#include <omp.h>
#define SIZE 100000
int arr[SIZE];
int arrcopy[SIZE];
void swap(int *a, int *b){
int t = *a;
*a = *b;
*b = t;
}
int partition_serial(int * arr, int low, int high){
int pivot = arr[high];
int i = (low - 1);
for(int j = low; j <= high - 1; j++){
if(arr[j] <= pivot){
i++;
swap(&arr[i], &arr[j]);
}
}
swap(&arr[i+1], &arr[high]);
return (i + 1);
}
void quicksort_serial(int *a, int low, int high){
if(low < high){ 
int div = partition_serial(a, low, high); 
quicksort_serial(a, low, div - 1); 
quicksort_serial(a, div + 1, high); 
}
}
void quicksort_parellel(int *a, int low, int high, int threads){
if(threads == 1){
int thread_num = omp_get_thread_num();
int cpu_num = sched_getcpu();
printf("Range [%8d %8d] executed by thread %3d is running on CPU %3d\n", low, high, thread_num, cpu_num);
quicksort_serial(a, low, high);
}
else{
if(low < high){
int div = partition_serial(a, low, high);
#pragma omp parallel sections
{
#pragma omp section
quicksort_parellel(a, low, div - 1, threads/2);
#pragma omp section
quicksort_parellel(a, div + 1, high, threads - threads/2);
}
}
}
}
int main(int argc, char *argv[]){
if(argc < 2){
printf("Usage ./a.out <number_of_threads>\n");
exit(1);
}
int nthreads;
unsigned int thread_qty = atoi(argv[1]);
omp_set_num_threads(thread_qty);
int i;
int sz = SIZE;
double start_time, run_time;
srand(5); 
for (i=0; i<sz; i++){
arr[i] = 1+(rand()%sz);
arrcopy[i] = arr[i];
}
int threads =  omp_get_num_threads();
start_time = omp_get_wtime();
quicksort_parellel(arrcopy, 0, sz-1, threads);
run_time = omp_get_wtime() - start_time;
printf("%f\n", run_time);
for (i = 1; i < SIZE; i++)
{
if (!(arrcopy[i - 1] <= arrcopy[i]))
{
printf ("Implementation error parallelel: arr[%d]=%d > arr[%d]=%d\n", i - 1,
arr[i - 1], i, arr[i]);
return 1;
}
}
return 0;
}
