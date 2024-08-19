#include <math.h>
#include <malloc.h>
#include <stdlib.h>
#include <stdlib.h>
void __attribute__((noinline)) test_vec(float *x, float *y)
{
int j;
#pragma omp simd
for (j=0; j<4; j++)
{
y[j] = sinf(x[j]);
}
#pragma omp simd
for (j=4; j<8; j++)
{
y[j] = expf(x[j]);
}
#pragma omp simd
for (j=8; j<12; j++)
{
y[j] = logf(x[j]);
}
#pragma omp simd
for (j=12; j<16; j++)
{
y[j] = fabsf(x[j]);
}
#pragma omp simd
for (j=16; j<20; j++)
{
y[j] = sqrtf(x[j]);
}
}
void __attribute__((noinline)) test_sc(float *x, float *y)
{
int j;
for (j=0; j<4; j++)
{
y[j] = sinf(x[j]);
}
for (j=4; j<8; j++)
{
y[j] = expf(x[j]);
}
for (j=8; j<12; j++)
{
y[j] = logf(x[j]);
}
for (j=12; j<16; j++)
{
y[j] = fabsf(x[j]);
}
for (j=16; j<20; j++)
{
y[j] = sqrtf(x[j]);
}
}
int main (int argc, char* argv[])
{
const int N = 10 * 4;
float* input, *output, *input_sc, *output_sc;
if(posix_memalign((void **) &input, 64, N * sizeof(float)) != 0)
{
exit(1);
}
if(posix_memalign((void **) &output, 64, N * sizeof(float)) != 0)
{
exit(1);
}
if(posix_memalign((void **) &input_sc, 64, N * sizeof(float)) != 0)
{
exit(1);
}
if(posix_memalign((void **) &output_sc, 64, N * sizeof(float)) != 0)
{
exit(1);
}
int i;
for (i=0; i<N; i++)
{
input[i] = (i*0.9f)/(i+1);
input_sc[i] = (i*0.9f)/(i+1);
}
test_vec(input, output);
test_sc(input_sc, output_sc);
#define ERROR 0.01
for (i=0; i<N; i++)
{
if(fabsf(input_sc[i] - input[i]) > ERROR)
{
printf("ERROR: input_sc[%d] = %f != input[%d] = %f\n", i, input_sc[i], i, input[i]);
exit(1);
}
if(fabsf(output_sc[i] - output[i]) > ERROR)
{
printf("ERROR: output_sc[%d] = %f != output[%d] = %f\n", i, output_sc[i], i, output[i]);
exit(1);
}
}
printf("SUCCESS\n");
return 0;
}
