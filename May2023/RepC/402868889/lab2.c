#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv) {
const int count = 10000000;
const int amount = 100;
const int max_threads = 16;
const int random_seed = 920214;
int *array = 0;
int target = 16;
int index = -1;
int label = 0;
double start_time, average_time, end_time;
array = (int *)malloc(count * sizeof(int));
srand(random_seed);
target = rand() % (count * 10);
FILE *stream;
stream = fopen("lab2.txt", "w+");
for (int threads = 1; threads <= max_threads; threads++) {
srand(random_seed + threads);
average_time = 0;
index = count + 1;
for (int i = 0; i < amount; i++) {
for (int j = 0; j < count; j++) {
array[j] = rand() % (count * 10);
}
label = 0;
start_time = omp_get_wtime();
#pragma omp parallel num_threads(threads) shared(array, count, target, index, threads) default(shared)
{
label = 0;
#pragma omp for
for (int k = 0; k < threads; k++) {
for (int i = 0; i < count / threads; i++) {
if (label) {
break;
}
if (array[k * count / threads + i] == target) {
index = k * count / threads + i;
label = 1;
}
}
}
#pragma omp single
{
if (!label) {
for (int j = 0; j < count % threads; j++) {
if (array[threads * (count / threads) + i] == target) {
index = threads * (count / threads) + i;
label = 1;
}
}
}
}
}
end_time = omp_get_wtime();
average_time += end_time - start_time;
}
printf("===================================================\ntarget value: %d || occurrence index: %d\n===================================================\n", target, index);
average_time = average_time / amount;
fprintf(stream, "===============================\nThread amount: %d\nAverage time elapsed: %0.7lf\n===============================\n", threads, average_time);
}
fclose(stream);
return 0;
}