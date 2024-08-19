#include <stdio.h>
#include <stdlib.h>
#define VECTOR_SIZE 64
void __attribute__((noinline)) saxpy(float *x, float *y, float *z, float a, int N)
{
#pragma omp parallel firstprivate(x, y, z, a, N)
{
int j;
x;
#pragma omp simd for schedule(dynamic)
for (j=0; j<N; j+=1)
{
z[j] = a * x[j] + y[j];
}
}
printf("\n");
}
int main (int argc, char * argv[])
{
const int N = 244 * 16 + 15;
const int iters = 1;
float *x, *y, *z; 
posix_memalign((void **)&x, VECTOR_SIZE, N*sizeof(float));
posix_memalign((void **)&y, VECTOR_SIZE, N*sizeof(float));
posix_memalign((void **)&z, VECTOR_SIZE, N*sizeof(float));
float a = 0.93f;
int i, j;
for (i=0; i<N; i++)
{
x[i] = i+1;
y[i] = i-1;
z[i] = 0.0f;
}
for (i=0; i<iters; i++)
{
saxpy(x, y, z, a, N);
}
for (i=0; i<N; i++)
{
if (z[i] != (a * x[i] + y[i]))
{
printf("Error\n");
return (1);
}
}
printf("SUCCESS!\n");
return 0;
}
