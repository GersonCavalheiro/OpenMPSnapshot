#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#define N 50000
#define UPPER N*4
#define LOWER 1
void count_sort(int a[], int n, int thread_count);
void print_array(int a[], int n);
void init_array(int array[], int n, int upper, int lower);
void display_time(double start, double end);
int main(int argc, char* argv[]) {
int thread_count, n;
if(argc != 3){
printf("Usage: <thread_count> <n>");
}
thread_count = strtoll(argv[1],NULL,10);
n = strtoll(argv[2],NULL,10);
int array[n];
init_array(array, n, UPPER, LOWER);
(void) printf("Sorting began...\n\n");
double start = omp_get_wtime();
count_sort(array, n, thread_count);
double end = omp_get_wtime();
display_time(start, end);
return 0;
return 0;
}
void count_sort(int a[], int n, int thread_count) {
int i, j, count;
int* temp = malloc(n*sizeof(int));
#pragma omp parallel for num_threads(thread_count) private(i,j,count) shared(a, temp)
for(i=0;i<n;i++) {
count = 0;
for (j = 0; j < n; j++)
if ((a[j] < a[i]) || (a[j] == a[i] &&  i< j))
count++;
temp[count] = a[i];
}
memcpy(a, temp, n*sizeof(int));
free(temp);
}
void print_array(int a[], int n) {
int i;
printf("[ ");
for(i=0;i<n;i++) {
printf("%d ", a[i]);
}
printf("] \n\n");
}
void init_array(int array[], int n, int upper, int lower){
int i;    
for (i=0; i<n; ++i)
array[i] = (rand() % (upper - lower + 1)) + lower;
}
void display_time(double start, double end){
(void) printf("Time spent for sorting: %f seconds\n", (end-start));
}
