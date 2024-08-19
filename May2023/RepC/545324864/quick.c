#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#define DEBUG 0
#ifndef THREADS
#define THREADS 1
#endif
#ifndef DIM
#define DIM 10
#endif
FILE *fp;
int a_seq[DIM];
int a_par[DIM];
void initialize();
void quick_sequential();
void quick_parallel();
int test_equality();
static int chunksize = 10;
int main(void)
{
omp_set_num_threads(THREADS);
printf("! Utilizing %d threads over %d elements\n", THREADS, DIM);
initialize();
double start_time, time;
start_time = omp_get_wtime();
quick_sequential(a_seq, 0, DIM - 1);
time = omp_get_wtime() - start_time;
printf("! Executed sequential quicksort: %f\n", time);
start_time = omp_get_wtime();
chunksize = DIM / THREADS;
#pragma omp parallel
{
#pragma omp single
quick_parallel(a_par, 0, DIM - 1);
}
time = omp_get_wtime() - start_time;
printf("! Executed parallelized quicksort: %f\n", time);
int res = test_equality();
printf("! Quicksort exit: %d\n", res);
if (res != 0)
{
fp = fopen("log/c_std.csv", "a");
fprintf(fp, "Quicksort error: Parallel and Sequential results differ.");
fclose(fp);
}
else
{
fp = fopen("log/c_std.csv", "a");
fprintf(fp, "Quicksort, %d, %d, %f\n", DIM, THREADS, time);
fclose(fp);
}
return 0;
}
void initialize()
{
if (DEBUG)
printf("\tInitializing arrays\n");
for (int i = 0; i < DIM; i++)
{
int r = rand() % 20 + 1; 
a_seq[i] = r;
a_par[i] = r;
if (DEBUG)
printf("\tOverwriting arrays with %d at %d\n", i, r);
}
}
void swap(int *a, int *b)
{
int t = *a;
*a = *b;
*b = t;
}
int partition(int arr[], int low, int high)
{
int pivot = arr[high];
int i = low - 1;
for (int j = low; j <= high - 1; j++)
{
if (arr[j] <= pivot)
{
i++;
swap(&arr[i], &arr[j]);
}
}
swap(&arr[i + 1], &arr[high]);
return i + 1;
}
void quick_sequential(int arr[], int low, int high)
{
if (low < high)
{
int pt = partition(arr, low, high);
quick_sequential(arr, low, pt - 1);
quick_sequential(arr, pt + 1, high);
}
}
void quick_parallel(int arr[], int low, int high)
{
if (low < high)
{
int pt = partition(arr, low, high);
#pragma omp task shared(arr) 
quick_parallel(arr, low, pt - 1);
#pragma omp task shared(arr) 
quick_parallel(arr, pt + 1, high);
}
}
int test_equality()
{
if (DEBUG)
printf("! Asserting equality of results\n");
if (DEBUG)
printf("\ta_seq\t|\ta_par\n");
for (int i = 0; i < DIM; i++)
{
if (DEBUG)
printf("%d\t%d\t|\t %d\n", i, a_seq[i], a_par[i]);
if (a_seq[i] != a_par[i])
{
return 1;
}
}
return 0;
}