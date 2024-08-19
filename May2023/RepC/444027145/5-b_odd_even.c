#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
const int RMAX = 100;
void odd_even_sort(int a[], int n, int);
int main(int argc, char* argv[]) {
int thread_count = 4;
int n = 20;
if (argc > 2){
thread_count = atoi(argv[1]);
n = atoi(argv[2]);
}
int* a = malloc(n * sizeof(int));
srand(0);
for (int i = 0; i < n; i++)
a[i] = rand() % RMAX;
if (n <= 20) {
for (int i = 0; i < n; i++)
printf("%2d ", a[i]);
printf("before\n");
}
double t0 = omp_get_wtime();
odd_even_sort(a, n, thread_count);
double t1 = omp_get_wtime();
printf("time = %.3f sec\n", t1 - t0);
if (n <= 20) {
for (int i = 0; i < n; i++)
printf("%2d ", a[i]);
printf("after\n");
}
free(a);
return 0;
}
void odd_even_sort(int *a, int n, int thread_count) {
int phase, i, temp;
default(none) shared(a, n) private(i, temp, phase)
for (phase = 0; phase < n; phase++) 
if (phase % 2 == 0) {   
#pragma omp parallel for num_threads(thread_count) default(none) shared(a, n) private(i, temp)
for (i = 1; i < n; i += 2) 
if (a[i-1] > a[i]) {
temp = a[i];
a[i] = a[i-1];
a[i-1] = temp;
}
} else {                
#pragma omp parallel for num_threads(thread_count) default(none) shared(a, n) private(i, temp)
for (i = 1; i < n-1; i += 2)
if (a[i] > a[i+1]) {
temp = a[i];
a[i] = a[i+1];
a[i+1] = temp;
}
}
}
