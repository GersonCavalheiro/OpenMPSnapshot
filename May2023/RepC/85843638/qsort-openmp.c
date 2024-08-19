#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define CHUNK_SIZE 6
#define THREAD_NUM 0
#define ARRAY_SIZE 10
#define DEBUG 1
int compare(const void *a, const void *b){
return (*(int*)b - *(int*)a);
}
void qsort_parallel(){
int a[ARRAY_SIZE] = {8, 5, 2, 6, 8, 7, 9, 2, 1, 4};
int num_t = 2;
omp_set_dynamic(0);
omp_set_num_threads(2);
#pragma omp for schedule(static)
for(int i=0; i<num_t; i++){
qsort((a + CHUNK_SIZE * i), CHUNK_SIZE, sizeof(int), compare);
}
qsort(a, ARRAY_SIZE, sizeof(int), compare);
}
void qsort_n(){
int a[ARRAY_SIZE] = {8, 5, 2, 6, 8, 7, 9, 2, 1, 4};
qsort(a, ARRAY_SIZE, sizeof(int), compare);
}