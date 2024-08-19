#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define VECTOR_SIZE 64
#define INIT_VALUE 2.0f
void __attribute__((noinline)) moving_average(
float * __restrict__ __attribute__((__aligned__(VECTOR_SIZE))) b,
float * __restrict__ __attribute__((__aligned__(VECTOR_SIZE))) c,
int N, int points)
{
int i, j;
#pragma omp simd for aligned(b,c:64) suitable(N) nowait overlap(b:4,0,0) 
for(i=0; i<(N-points); i++)
{
float tmp = 0.0f;        
for(j=0; j<points; j++)
{
tmp += b[i+j];
}
c[i] = tmp / points;
}
}
void __attribute__((noinline)) moving_average_sc(
float * __restrict__ __attribute__((__aligned__(VECTOR_SIZE))) b,
float * __restrict__ __attribute__((__aligned__(VECTOR_SIZE))) c,
int N, int points)
{
int i, j;
for(i=0; i<(N-points); i++)
{
float tmp = 0.0f;        
for(j=0; j<points; j++)
{
tmp += b[i+j];
}
c[i] = tmp / points;
}
}
int main (int argc, char* argv[])
{
const int N  = 1000;
const int iters  = 1;
const int points = 70;
int i;
float* input, *result, *result_sc;
fprintf(stderr, "Init\n");
if(posix_memalign((void **) &input, VECTOR_SIZE, N * sizeof(float)) != 0)
{
exit(1);
}
if(posix_memalign((void **) &result, VECTOR_SIZE, N * sizeof(float)) != 0)
{
exit(1);
}
if(posix_memalign((void **) &result_sc, VECTOR_SIZE, N * sizeof(float)) != 0)
{
exit(1);
}
#pragma omp parallel for firstprivate(N) private(i)
for (i=0; i<N; i++)
{
input[i] = _popcnt32(i);
}
#pragma omp parallel private(i) firstprivate(iters, input, result, N, points)
{
for (i=0; i<iters; i++)
{
moving_average(input, result, N, points);
}
}
fprintf(stderr, "%.10f\n", result[N-1]);
moving_average_sc(input, result_sc, N, points);
#pragma omp parallel for firstprivate(N, points, input, result, result_sc)
for(i=0; i<N-points; i++)
{
if(fabs(result[i]-result_sc[i]) > 5)
{
printf("ERROR (%i) %.20f != %.20f\n", i, result[i], result_sc[i]);
exit(1);
}
}
printf("SUCCESS\n");
return 0;
}
