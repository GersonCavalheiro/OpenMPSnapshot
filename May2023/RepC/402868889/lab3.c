#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
void insertionSort(int *array, int count, int gap, int i) {
for (int j = gap + i; j < count; j += gap) {
while (j > i && array[j - gap] > array[j]) {
int temp = array[j];
array[j] = array[j - gap];
array[j - gap] = temp;
j -= gap;
}
}
}
void parallelShellSort(int *array, int count, int threads) {
for (int gap = count / 2; gap > 0; gap /= 2) {
#pragma omp parallel for num_threads(threads) shared(array, count, gap) default(none)
for (int i = 0; i < gap; i++) {
insertionSort(array, count, gap, i);
}
}
}
int main(int argc, char **argv) {
const int count = 100000;
const int amount = 100;
const int max_threads = 16;
const int random_seed = 9202;
int *array = 0;
double start_time, average_time, end_time;
array = (int *)malloc(count * sizeof(int));
FILE *stream;
stream = fopen("lab3.txt", "w+");
for (int threads = 1; threads <= max_threads; threads++) {
srand(random_seed + threads);
average_time = 0;
for (int i = 0; i < amount; i++) {
for (int j = 0; j < count; j++) {
array[j] = rand() % (count * 10);
}
start_time = omp_get_wtime();
parallelShellSort(array, count, threads);
end_time = omp_get_wtime();
average_time += end_time - start_time;
}
average_time = average_time / amount;
fprintf(stream, "===============================\nThread amount: %d\nAverage time elapsed: %0.7lf\n===============================\n", threads, average_time);
}
fclose(stream);
return 0;
}