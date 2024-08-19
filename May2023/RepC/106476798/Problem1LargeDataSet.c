#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
#include <sys/time.h>
#include <omp.h>
#define NROW 1768149
#define ARRAY_SIZE 1000
double duration(struct timeval t0, struct timeval t1)
{
return (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_usec - t0.tv_usec) / 1000.0;
}
long combine_into_store(int old_value, int new_value, int size) {
int *array = (int *)malloc(sizeof(int) * size);
array[0] = old_value;
array[1] = new_value;
array[2] = -1;
return -((long)array);
}
bool find_and_erase_value_in_store(long store, int value) {
int *array = (int *)(-store);
for (int i = 0; array[i] != -1; i++) {
if (array[i] == value) {
array[i] = 0;
return true;
}
}
return false;
}
void insert_value_into_store(long store, int value) {
int *array = (int *)(-store);
for (int i = 0; true; i++) {
if (array[i] == 0) {
array[i] = value;
break;
} else if (array[i] == -1) {
array[i] = value;
array[i + 1] = -1;
break;
}
}
}
long reduce_store_to_value_if_possible(long store) {
int value = 0;
int *array = (int *)(-store);
for (int i = 0; array[i] != -1; i++) {
if (array[i] > 0) {
if (value > 0) { 
return store;
} else {
value = array[i];
}
}
}
free(array);
return value;
}
bool find_and_erase_value_in_map(long *map, int value, int index) {
long store = map[index];
if (store > 0) {
if (store == value) {
map[index] = 0;
return true;
} else {
return false;
}
} else if (store < 0) {
if (find_and_erase_value_in_store(store, value)) {
map[index] = reduce_store_to_value_if_possible(store);
return true;
} else {
return false;
}
} else {
return false;
}
}
void insert_value_into_map(long *map, int value, int index, int max_array_size) {
long store = map[index];
if (store == 0) {
map[index] = value;
} else if (store > 0) {
map[index] = combine_into_store((int)store, value, max_array_size);
} else if (store < 0) {
insert_value_into_store(store, value);
map[index] = reduce_store_to_value_if_possible(store);
}
}
void insert_store_into_map(long *map, long store, int index, int max_array_size) {
int *array = (int *)(-store);
for (int i = 0; array[i] != -1; i++) {
int value = array[i];
if (value > 0) {
insert_value_into_map(map, value, index, max_array_size);
}
}
}
int recippar(int *edges, int nrow) {
int count = 0;
long *map = calloc(INT_MAX, sizeof(long));
#pragma omp parallel
{
int thread_num = omp_get_thread_num();
int nth = omp_get_num_threads();
int local_count = 0;
long *local_map = calloc(INT_MAX, sizeof(long));
for (int i = thread_num * 2; i < nrow * 2; i += nth * 2) {
int first = edges[i], second = edges[i + 1];
long store = local_map[second];
if (store == first) { 
local_map[second] = 0;
local_count += 1;
} else if (store >= 0) { 
insert_value_into_map(local_map, second, first, ARRAY_SIZE);
} else if (store < 0) { 
if (find_and_erase_value_in_store(store, first)) {
local_count += 1;
} else {
insert_value_into_map(local_map, second, first, ARRAY_SIZE);
}
}
}
for (int i = i; i < INT_MAX; i++) {
long store = local_map[i];
if (store > 0) { 
#pragma omp critical
{
insert_value_into_map(map, store, i, ARRAY_SIZE * nth);
}
} else if (store < 0) { 
#pragma omp critical
{
insert_store_into_map(map, store, i, ARRAY_SIZE * nth);
}
int *array = (int *)(-store);
free(array);
}
}
free(local_map);
#pragma omp critical
{
count += local_count;
}
}
#pragma omp barrier
{
for (int i = 1; i < INT_MAX; i++) {
long store = map[i];
if (store > 0 && store != i) { 
if (find_and_erase_value_in_map(map, i, store)) {
count += 1;
}
} else if (store < 0) { 
int *values = (int *)(-store);
for (int index = 0; values[index] != -1; index++) {
if (values[index] > 0 && values[index] != i) {
if (find_and_erase_value_in_map(map, i, values[index])) {
count += 1;
}
}
}
}
}
return count;
}
}
int main() {
FILE *twitter_combined = fopen("twitter_combined.txt", "r");
int *edges = (int *)malloc(sizeof(int) * NROW * 2);
for (int i = 0; i < NROW * 2; i++) {
fscanf(twitter_combined, "%d", &edges[i]);
}
struct timeval start;
gettimeofday(&start, NULL);
int count = recippar(edges, NROW);
struct timeval end;
gettimeofday(&end, NULL);
printf("Count: %d\nDuration: %lf\n", count, duration(start, end));
return 0;
}
