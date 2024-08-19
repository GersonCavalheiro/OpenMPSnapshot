

#define _CRT_SECURE_NO_WARNINGS
#define NUM_THREADS 8

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>

#define MAX(A, B) ((A)>(B))?(A):(B)
#define MIN(A, B) ((A)<(B))?(A):(B)

void printArray(int *array, int size);
void fillArray(int *array, int size);
void merge(int *a, int n, int m);
void mergeSort(int *a, int n);


int main(int argc, char *argv[]) {
int *array = NULL;
int size = 0;
if (argc < 2) {
printf("[-] Invalid No. of arguments.\n");
printf("[-] Try -> <size_of_array>\n");
printf(">>> ");
scanf("%d", &size);
}
else {
size = atoi(argv[1]);
}
array = (int *)malloc(sizeof(int) * size);

int NUM_RUNS = 2;
double TOTAL_TIME = 0.0;

fillArray(array, size);

omp_set_num_threads(NUM_THREADS);
printf("\n=======================================\n");
printf("\n[INFO] Computing with [%d] thread(s)...\n", NUM_THREADS);
printf("\n=======================================\n");

for(int i = 0; i < NUM_RUNS; i++) {


printf("Merge Sort:\n");

double start_time = omp_get_wtime();


#pragma omp parallel
{
#pragma omp single
mergeSort(array, size);
}

double elapsed_time = omp_get_wtime() - start_time;
TOTAL_TIME += elapsed_time;

}
printf("\n[INFO] Completed running with [%d] thread(s).", NUM_THREADS);
printf("\n[INFO] Average run time[6 Iterations] with [%d] thread(s): %lf\n", NUM_THREADS, TOTAL_TIME / NUM_RUNS);


free(array);

system("PAUSE");
return EXIT_SUCCESS;
}

void fillArray(int *array, int size) {
srand(time(NULL));
while (size-->0) {
*array++ = rand() % 100;
}
}

void printArray(int *array, int size) {
while (size-->0) {
printf("%d, ", *array++);
}
printf("\n");
}

void merge(int *a, int n, int m) {
int i, j, k;
int *temp = (int *)malloc(n * sizeof(int));
for (i = 0, j = m, k = 0; k < n; k++) {
temp[k] = j == n ? a[i++]
: i == m ? a[j++]
: a[j] < a[i] ? a[j++]
: a[i++];
}
for (i = 0; i < n; i++) {
a[i] = temp[i];
}
free(temp);
}

void mergeSort(int *a, int n) {

int m;
if(n < 2)
return;

m = n / 2;

#pragma omp task
mergeSort(a, m);

#pragma omp task
mergeSort(a + m, n - m);

#pragma omp taskwait
merge(a, n, m);
}
