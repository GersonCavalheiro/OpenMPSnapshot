#include <stdio.h>
#define VECTOR_SIZE 64
void foo_simd(float * __restrict__ __attribute__((__aligned__(VECTOR_SIZE))) b,
int N)
{
int i, j;
#pragma omp simd aligned(b:64)
for(i=N-16; i<N; i++)
{
int j;
float tmp = 0.0f;
for(j=0; j<(N-i); j+=1)
{
tmp += b[i+j];
}
b[i] = tmp / (N-i); 
}
}
void foo_scalar(float * __restrict__ __attribute__((__aligned__(VECTOR_SIZE))) b,
int N)
{
int i, j;
#pragma novector
for(i=N-16; i<N; i++)
{
int j;
float tmp = 0.0f;
for(j=0; j<(N-i); j+=1)
{
tmp += b[i+j];
}
b[i] = tmp / (N-i);
}
}
int main()
{
float __attribute__((__aligned__(VECTOR_SIZE))) a[64];
float __attribute__((__aligned__(VECTOR_SIZE))) b[64];
int N = 19;
int i;
for(i=0; i<64; i++)
{
a[i] = b[i] = i;
}
foo_simd(a, N);
foo_scalar(b, N);
for(i=0; i<64; i++)
{
if(a[i] != b[i])
{
printf("Error at %d: %f != %f\n", i, a[i], b[i]);
return 1;
}
}
printf("SUCCESS!\n");
return 0;
}
