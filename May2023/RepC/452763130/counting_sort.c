#include "counting_sort.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <stdlib.h>
#include "util.h"
static void min_max(const int *array, long long size, int *min, int *max,
int nthreads)
{
*min = array[0];
*max = array[0];
long long i = 0;
#pragma omp parallel for num_threads(nthreads) default(shared) private(i)
for (i = 0; i < size; i++) {
if (array[i] < *min) {
#pragma omp critical
{
if (array[i] < *min)
*min = array[i];
}
}
else if (array[i] > *max) {
#pragma omp critical
{
if (array[i] > *max)
*max = array[i];
}
}
}
}
static long key(int item, long long max) {
return item;
}
void counting_sort(int *array, long long size, int nthreads) {
long long i = 0, j = 0, k = 0;
int max = 0, min = 0;
min_max(array, size, &min, &max, nthreads);
long long count_size = max - min + 1;
int *count = (int *)safe_alloc(sizeof(int) * count_size);
#pragma omp parallel for num_threads(nthreads) shared(count, count_size) private(i)
for (i = 0; i < count_size; i++)
count[i] = 0;
#pragma omp parallel for num_threads(nthreads) shared(array, min, size) private(i, j) reduction(+: count[:count_size])
for (i = 0; i < size; i++) {
j = key(array[i], size) - min;
count[j] += 1;
}
for (i = min; i <= max; i++)
for (j = 0; j < count[i - min]; j++)
array[k++] = i;
free(count);
}
