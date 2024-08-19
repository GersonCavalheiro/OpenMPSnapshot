#include <omp.h>
#include <stdio.h>
void ParallelCopy(const int* a, int* b, int size, int threadCount)
{
int i;
#pragma omp parallel for num_threads(threadCount)
for (i = 0; i < size; ++i)
{
b[i] = a[i];
}
}
int main(void)
{
int array[10] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
int result[10];
ParallelCopy(array, result, 10, 8);
return 0;
}