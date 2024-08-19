#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <pthread.h>
#include "bucketsort.h"
int *bucket_sort_openmp(int *array, size_t size, int max_value, int n_threads){
Bucket *buckets = NULL;
int *sort_array = NULL;
long double start, final;
start = omp_get_wtime();
buckets = allocate_buckets(n_threads);
sort_array = malloc(size * sizeof(int));
for(size_t i = 0; i < size; i++){
int index = get_bucket_index(array[i],max_value,n_threads);
buckets[index] = add_bucket_value(buckets[index], array[i]);
}
omp_set_dynamic(0);
#pragma omp parallel num_threads(n_threads)
{
#pragma omp parallel for
for(int i = 0; i < n_threads; i++)
bubble_sort((void *) &buckets[i]);
}
combine_buckets(sort_array, buckets, n_threads);
final = omp_get_wtime() - start;
printf("| | | utilizando %d thread(s): %Lf\n", n_threads, final);
free_buckets(buckets, n_threads);
return sort_array;
}
int *bucket_sort_pthreads(int *array, size_t size, int max_value, int n_threads){
pthread_t threads[n_threads];
Bucket *buckets = NULL;
int *sort_array = NULL;
long double start, final;
start = omp_get_wtime();
buckets = allocate_buckets(n_threads);
sort_array = malloc(size * sizeof(int));
for(size_t i = 0; i < size; i++){
int index = get_bucket_index(array[i],max_value,n_threads);
buckets[index] = add_bucket_value(buckets[index], array[i]);
}
for(int i = 0; i < n_threads; i++)
pthread_create(&threads[i], NULL, (void *) bubble_sort, (void *) &buckets[i]);
for(int i = 0; i < n_threads; i++){
pthread_join(threads[i], NULL);
}
combine_buckets(sort_array, buckets, n_threads);
final = omp_get_wtime() - start;
printf("| | | utilizando %d thread(s): %Lf\n", n_threads, final);
free_buckets(buckets, n_threads);
return sort_array;
}
Bucket *allocate_buckets(int n_arrays){
Bucket *arrays;
arrays = (Bucket *) malloc(n_arrays * sizeof(Bucket));
for(int i = 0; i < n_arrays; i++){
arrays[i].array = NULL;
arrays[i].size = 0;
}
return arrays;
}
int get_bucket_index(int value, int max, int n_buckets){
return (int) (value)/(max/n_buckets);
}
Bucket add_bucket_value(Bucket array, int value){
array.array = realloc(array.array, (array.size + 1) * sizeof(int));
array.array[array.size] = value;
array.size++;
return array;
}
void bubble_sort(void *array){
Bucket *sort = (Bucket *) array;
char flag;
int temp;
for(size_t i = 0; i < sort->size; ++i){
flag = 1;
for(size_t j = 0; j < sort->size - 1 - i; ++j){
if(sort->array[j] > sort->array[j+1]){
flag = 0;
temp = sort->array[j];
sort->array[j] = sort->array[j+1];
sort->array[j+1] = temp;
}
}
if(flag)
return;
}
return;
}
void combine_buckets(int *sort_array, Bucket *buckets, int n_buckets){
int cont = 0;
for(int i = 0; i < n_buckets; i++){
for(size_t j = 0; j < buckets[i].size; j++)
sort_array[cont++] = buckets[i].array[j];
}
}
void free_buckets(Bucket *buckets, int n_buckets){
for(int i = 0; i < n_buckets; i++)
free(buckets[i].array);
free(buckets);
}
