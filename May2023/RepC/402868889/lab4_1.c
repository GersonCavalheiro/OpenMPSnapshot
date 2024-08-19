#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
omp_lock_t lock;
void shellSort(int *array, int count, int threads) {
omp_init_lock(&lock);
for (int gap = count / 2; gap > 0; gap /= 2) {
#pragma omp parallel for num_threads(threads) shared(lock, count, gap) default(shared)
for (int i = gap; i < count; i += 1) {
int temp = array[i];
int j;
for (j = i; j >= gap && array[j - gap] > temp; j -= gap) {
array[j] = array[j - gap];
}
omp_set_lock(&lock);
array[j] = temp;
omp_unset_lock(&lock);
}
}
}
int main(int argc, char **argv) {
const int count = 10;
const int threads = 16;
const int random_seed = 100;
int *array = 0;
srand(random_seed);
array = (int *)malloc(count * sizeof(int));
for (int i = 0; i < count; i++) {
array[i] = rand();
}
shellSort(array, count, threads);
for (int i = 0; i < count; i++) {
printf("array[%d] = %d\n", i, array[i]);
}
free(array);
omp_destroy_lock(&lock);
return 0;
}