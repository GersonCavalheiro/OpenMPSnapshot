#ifndef __ICC
#include <stdio.h>
#include <stdlib.h>
enum { NUM_ITEMS = 500, ARRAY_SIZE = 1000 };
void f1(int array_size, int (*a)[array_size], int *result)
{
int i;
for (i = 0; i < array_size; i++)
{
result[i] = 0;
}
#pragma omp parallel for firstprivate(a) reduction(+:[array_size]result)
for (i = 0; i < NUM_ITEMS; i++)
{
int j;
for (j = 0; j < array_size; j++)
{
result[j] += a[i][j];
}
}
}
enum { RESULT = 124750 };
int main(int argc, char *argv[])
{
int a[NUM_ITEMS][ARRAY_SIZE];
int result[ARRAY_SIZE];
int i, j;
for (i = 0; i < NUM_ITEMS; i++)
{
for (j = 0; j < ARRAY_SIZE; j++)
{
a[i][j] = i;
}
}
for (j = 0; j < ARRAY_SIZE; j++)
{
result[j] = 0;
}
f1(ARRAY_SIZE, a, result);
for (j = 0; j < ARRAY_SIZE; j++)
{
if (result[j] != RESULT)
{
fprintf(stderr, "wrong result[%d] -> %d != %d\n", j, result[j], RESULT);
abort();
}
}
return 0;
}
#else
int main(int argc, char *argv[])
{
return 0;
}
#endif
