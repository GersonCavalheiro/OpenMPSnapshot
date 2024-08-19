#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "tris.h"
#define CHUNK_SIZE 6
#define TASKS_NUM 2
#define ARRAY_SIZE 10
void merge_sort(int a[], int ig, int id){
if(ig < id){
int m = (ig + id)/2;
merge_sort(a, ig, m);
merge_sort(a, m+1, id);
merge(a, ig, m, id, m);
}
}
void merge_sort_omp(int a[], int ig, int id){
if(ig < id){
int m = (ig + id)/2;
#pragma omp task
merge_sort_omp(a, ig, m);
#pragma omp task
merge_sort_omp(a, m+1, id);
#pragma omp wait
merge(a, ig, m, id, m);
}
}
