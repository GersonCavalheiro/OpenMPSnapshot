#include <stdio.h>
#include <stdlib.h>
#define VECTOR_SIZE 64
void test(void * z, int N)
{
int *_z = (int *) z;
int i;
for (i=0; i<N; i++)
{
if (_z[i] == 1)
{
printf("Error\n");
exit (1);
}
}
}
void __attribute__((noinline)) lt_int(int *x, int *y, int *z, int N)
{
int j;
#pragma omp simd
for (j=0; j<N; j++)
{
z[j] = (x[j] < y[j]) ? 0 : 1;
}
}
void __attribute__((noinline)) le_int(int *x, int *y, int *z, int N)
{
int j;
#pragma omp simd
for (j=0; j<N; j++)
{
z[j] = (x[j] <= y[j]) ? 0 : 1;
}
}
void __attribute__((noinline)) gt_int(int *x, int *y, int *z, int N)
{
int j;
#pragma omp simd
for (j=0; j<N; j++)
{
z[j] = (x[j] > y[j]) ? 0 : 1;
}
}
void __attribute__((noinline)) ge_int(int *x, int *y, int *z, int N)
{
int j;
#pragma omp simd
for (j=0; j<N; j++)
{
z[j] = (x[j] >= y[j]) ? 0 : 1;
}
}
void __attribute__((noinline)) eq_int(int *x, int *y, int *z, int N)
{
int j;
#pragma omp simd
for (j=0; j<N; j++)
{
z[j] = (x[j] == y[j]) ? 0 : 1;
}
}
void __attribute__((noinline)) diff_int(int *x, int *y, int *z, int N)
{
int j;
#pragma omp simd
for (j=0; j<N; j++)
{
z[j] = (x[j] != y[j]) ? 0 : 1;
}
}
int main (int argc, char * argv[])
{
const int N = 16;
const int iters = 1;
int *x, *y, *z;
posix_memalign((void **)&x, VECTOR_SIZE, N*sizeof(int));
posix_memalign((void **)&y, VECTOR_SIZE, N*sizeof(int));
posix_memalign((void **)&z, VECTOR_SIZE, N*sizeof(int));
int i, j;
for (i=0; i<N; i++)
{
x[i] = i;
y[i] = i+1;
z[i] = 0.0f;
}
lt_int(x, y, z, N);
test((void *)z, N);
gt_int(y, x, z, N);
test((void *)z, N);
le_int(x, y, z, N);
test((void *)z, N);
ge_int(y, x, z, N);
test((void *)z, N);
diff_int(y, x, z, N);
test((void *)z, N);
for (i=0; i<N; i++)
{
x[i] = i;
y[i] = i;
}
eq_int(y, x, z, N);
test((void *)z, N);
printf("SUCCESS!\n");
return 0;
}
