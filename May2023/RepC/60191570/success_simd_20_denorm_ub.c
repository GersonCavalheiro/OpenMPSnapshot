#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define VECTOR_SIZE 64
typedef int * __restrict__ F;
__attribute__((noinline)) int foo(
int * __restrict__ __attribute__((__aligned__(VECTOR_SIZE))) b,
int N)
{
int i,j;
int tmp = 0.0f;
#pragma omp simd for simd_reduction(+:tmp) nowait
for(i=0; i<(N-16); i++)
{
tmp += b[i] *
(((b[i+1] +
b[i+2] *
b[i+3]) +
(b[i+4] +
b[i+5] *
b[i+6]) *
((b[i+7] +
b[i+8] *
b[i+9]) +
(b[i+10] +
b[i+11] *
b[i+12]) *
(b[i+13] +
b[i+14] *
b[i+15]))));
}
return tmp;
}
int main (int argc, char* argv[])
{
int* b;
int result, i;
int N = 16000;
if(posix_memalign((void **) &b, VECTOR_SIZE, N * sizeof(int)) != 0)
{
return 1;
}
for (i=0; i<N; i++)
{
b[i] = 5;
}
result = 0;
N=7;
for(i=0; i<N-16; i++)
{
result += b[i] *
(((b[i+1] +
b[i+2] *
b[i+3]) +
(b[i+4] +
b[i+5] *
b[i+6]) *
((b[i+7] +
b[i+8] *
b[i+9]) +
(b[i+10] +
b[i+11] *
b[i+12]) *
(b[i+13] +
b[i+14] *
b[i+15]))));
}
if(foo(b,N) != result)
{
printf("ERROR %d != %d\n", foo(b,N), result);
return 1;
}
printf("%d and %d\n",  foo(b,N), result);
printf("SUCCESS\n");
result = 0;
N = 1600;
for(i=0; i<N-16; i++)
{
result += b[i] *
(((b[i+1] +
b[i+2] *
b[i+3]) +
(b[i+4] +
b[i+5] *
b[i+6]) *
((b[i+7] +
b[i+8] *
b[i+9]) +
(b[i+10] +
b[i+11] *
b[i+12]) *
(b[i+13] +
b[i+14] *
b[i+15]))));
}
if(foo(b,N) != result)
{
printf("ERROR %d != %d\n", foo(b,N), result);
return 1;
}
return 0;
}
