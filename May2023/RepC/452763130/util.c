#include "util.h"
#include <stdlib.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#endif
void *safe_alloc(long long size) {
if (size < 1) {
fprintf(stderr, "Can not allocate memory of %lld bytes.\n", size);
exit(EXIT_FAILURE);
}
void *ptr = malloc(size);
if (ptr == NULL) {
fprintf(stderr, "Could not allocate memory of %lld bytes.\n", size);
exit(EXIT_FAILURE);
}
return ptr;
}
FILE *file_open(const char *path, const char *mode) {
FILE *f = fopen(path, mode);
if (f == NULL) {
fprintf(stderr, "Could not open file '%s'.\n", path);
exit(EXIT_FAILURE);
}
return f;
}
void array_init_random(int *array, long long size, int min, int max,
int nthreads)
{
long long i = 0;
#pragma omp parallel num_threads(nthreads) shared(array, size) private(i)
{
unsigned seed = time(NULL) ^ omp_get_thread_num();
#pragma omp for
for (i = 0; i < size; i++)
array[i] = rand_r(&seed) % (max + 1 - min) + min;
}
}
void array_show(const int *array, long long size) {
printf("----------------------- ARRAY OF %lld ELEMENTS:\n", size);
for (long long i = 0; i < size; i++) {
printf("%5d ", array[i]);
if (i > 0 && i % 10 == 0)
printf("\n");
}
printf("\n\n");
}
