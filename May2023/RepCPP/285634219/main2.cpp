





#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define ELEMENTARY_LOG2SIZE 11

extern"C" void fwtCPU(float *h_Output, float *h_Input, int log2N);
extern"C" void slowWTcpu(float *h_Output, float *h_Input, int log2N);
extern "C" void dyadicConvolutionCPU(
float *h_Result,
float *h_Data,
float *h_Kernel,
int log2dataN,
int log2kernelN
);


const int log2Data = 12;
const int dataN = 1 << log2Data;
const int DATA_SIZE = dataN * sizeof(float);

const int log2Kernel = 7;
const int kernelN = 1 << log2Kernel;
const int KERNEL_SIZE = kernelN * sizeof(float);




int main(int argc, char *argv[])
{
double delta, ref, sum_delta2, sum_ref2, L2norm;

int i;

printf("Data length: %i; kernel length: %i\n", dataN, kernelN);

printf("Initializing data...\n");
float *h_Kernel    = (float *)malloc(KERNEL_SIZE);
float *h_Data      = (float *)malloc(DATA_SIZE);
float *h_ResultCPU = (float *)malloc(DATA_SIZE);

srand(123);
for (i = 0; i < kernelN; i++)
{
h_Kernel[i] = (float)rand() / (float)RAND_MAX;
}

for (i = 0; i < dataN; i++)
{
h_Data[i] = (float)rand() / (float)RAND_MAX;
}

printf("Running GPU dyadic convolution using Fast Walsh Transform...\n");

float *d_Kernel = (float *)malloc(DATA_SIZE);
float *d_Data   = (float *)malloc(DATA_SIZE);

#pragma omp target data map (alloc: d_Kernel[0:dataN], d_Data[0:dataN])
{
for (i = 0; i < 1; i++)
{
memset(d_Kernel, 0, DATA_SIZE);
memcpy(d_Kernel, h_Kernel, KERNEL_SIZE);
#pragma omp target update to (d_Kernel[0:dataN])

memcpy(d_Data, h_Data, DATA_SIZE);
#pragma omp target update to (d_Data[0:dataN])

int log2N = log2Data;
int N = 1 << log2N;
int M = 1;

for (; log2N > ELEMENTARY_LOG2SIZE; log2N -= 2, N >>= 2, M <<= 2)
{
#pragma omp target teams distribute parallel for thread_limit(256)
for (int pos = 0; pos < N; pos++) {
const int stride = N/4;
int lo = pos & (stride - 1);
int i0 = ((pos - lo) << 2) + lo;
int i1 = i0 + stride;
int i2 = i1 + stride;
int i3 = i2 + stride;

float D0 = d_Data[i0];
float D1 = d_Data[i1];
float D2 = d_Data[i2];
float D3 = d_Data[i3];

float T;
T = D0;
D0        = D0 + D2;
D2        = T - D2;
T = D1;
D1        = D1 + D3;
D3        = T - D3;
T = D0;
d_Data[i0] = D0 + D1;
d_Data[i1] = T - D1;
T = D2;
d_Data[i2] = D2 + D3;
d_Data[i3] = T - D3;
}
}

#ifdef DEBUG
#pragma omp target update from (d_Data[0:dataN])
for (int i = 0; i < dataN; i++) printf("k1 %f\n", d_Data[i]);
#endif


#pragma omp target teams num_teams(M) thread_limit(N/4)
{
float s_data[2048];
#pragma omp parallel 
{
int lid = omp_get_thread_num();
int gid = omp_get_team_num();
int gsz = omp_get_num_threads(); 

const int    N = 1 << log2N;
const int base = gid << log2N;

const float *d_Src = d_Data  + base;
float *d_Dst = d_Data + base;

for (int pos = lid; pos < N; pos += gsz)
{
s_data[pos] = d_Src[pos];
}

#pragma omp barrier

const int pos = lid;

for (int stride = N >> 2; stride > 0; stride >>= 2)
{
int lo = pos & (stride - 1);
int i0 = ((pos - lo) << 2) + lo;
int i1 = i0 + stride;
int i2 = i1 + stride;
int i3 = i2 + stride;

float D0 = s_data[i0];
float D1 = s_data[i1];
float D2 = s_data[i2];
float D3 = s_data[i3];

float T;
T = D0;
D0         = D0 + D2;
D2         = T - D2;
T = D1;
D1         = D1 + D3;
D3         = T - D3;
T = D0;
s_data[i0] = D0 + D1;
s_data[i1] = T - D1;
T = D2;
s_data[i2] = D2 + D3;
s_data[i3] = T - D3;
#pragma omp barrier
}

if (log2N & 1)
{
#pragma omp barrier

for (int pos = lid; pos < N / 2; pos += gsz)
{
int i0 = pos << 1;
int i1 = i0 + 1;

float D0 = s_data[i0];
float D1 = s_data[i1];
s_data[i0] = D0 + D1;
s_data[i1] = D0 - D1;
}
}

#pragma omp barrier

for (int pos = lid; pos < N; pos += gsz)
{
d_Dst[pos] = s_data[pos];
}
}
}
#ifdef DEBUG
#pragma omp target update from (d_Data[0:dataN])
for (int i = 0; i < dataN; i++) printf("k2 %f\n", d_Data[i]);
#endif

log2N = log2Data;
N = 1 << log2N;
M = 1;

for (; log2N > ELEMENTARY_LOG2SIZE; log2N -= 2, N >>= 2, M <<= 2)
{
#pragma omp target teams distribute parallel for thread_limit(256)
for (int pos = 0; pos < N; pos++) {
const int stride = N/4;
int lo = pos & (stride - 1);
int i0 = ((pos - lo) << 2) + lo;
int i1 = i0 + stride;
int i2 = i1 + stride;
int i3 = i2 + stride;

float D0 = d_Kernel[i0];
float D1 = d_Kernel[i1];
float D2 = d_Kernel[i2];
float D3 = d_Kernel[i3];

float T;
T = D0;
D0        = D0 + D2;
D2        = T - D2;
T = D1;
D1        = D1 + D3;
D3        = T - D3;
T = D0;
d_Kernel[i0] = D0 + D1;
d_Kernel[i1] = T - D1;
T = D2;
d_Kernel[i2] = D2 + D3;
d_Kernel[i3] = T - D3;
}
}
#ifdef DEBUG
#pragma omp target update from (d_Kernel[0:dataN])
for (int i = 0; i < dataN; i++) printf("k3 %f\n", d_Kernel[i]);
#endif

#pragma omp target teams num_teams(M) thread_limit(N/4)
{
float s_data[2048];
#pragma omp parallel 
{
int lid = omp_get_thread_num();
int gid = omp_get_team_num();
int gsz = omp_get_num_threads(); 

const int    N = 1 << log2N;
const int base = gid << log2N;

const float *d_Src = d_Kernel  + base;
float *d_Dst = d_Kernel + base;

for (int pos = lid; pos < N; pos += gsz)
{
s_data[pos] = d_Src[pos];
}

const int pos = lid;

for (int stride = N >> 2; stride > 0; stride >>= 2)
{
int lo = pos & (stride - 1);
int i0 = ((pos - lo) << 2) + lo;
int i1 = i0 + stride;
int i2 = i1 + stride;
int i3 = i2 + stride;

#pragma omp barrier
float D0 = s_data[i0];
float D1 = s_data[i1];
float D2 = s_data[i2];
float D3 = s_data[i3];

float T;
T = D0;
D0         = D0 + D2;
D2         = T - D2;
T = D1;
D1         = D1 + D3;
D3         = T - D3;
T = D0;
s_data[i0] = D0 + D1;
s_data[i1] = T - D1;
T = D2;
s_data[i2] = D2 + D3;
s_data[i3] = T - D3;
}

if (log2N & 1)
{
#pragma omp barrier

for (int pos = lid; pos < N / 2; pos += gsz)
{
int i0 = pos << 1;
int i1 = i0 + 1;

float D0 = s_data[i0];
float D1 = s_data[i1];
s_data[i0] = D0 + D1;
s_data[i1] = D0 - D1;
}
}

#pragma omp barrier

for (int pos = lid; pos < N; pos += gsz)
{
d_Dst[pos] = s_data[pos];
}
}
}
#ifdef DEBUG
#pragma omp target update from (d_Kernel[0:dataN])
for (int i = 0; i < dataN; i++) printf("k4 %f\n", d_Kernel[i]);
#endif


float     rcpN = 1.0f / (float)dataN;
#pragma omp target teams distribute parallel for num_teams(128) thread_limit(256)
for (int pos = 0; pos < dataN; pos++)
{
d_Data[pos] *= d_Kernel[pos] * rcpN;
}

log2N = log2Data;
N = 1 << log2N;
M = 1;

for (; log2N > ELEMENTARY_LOG2SIZE; log2N -= 2, N >>= 2, M <<= 2)
{
#pragma omp target teams distribute parallel for thread_limit(256)
for (int pos = 0; pos < N; pos++) {
const int stride = N/4;
int lo = pos & (stride - 1);
int i0 = ((pos - lo) << 2) + lo;
int i1 = i0 + stride;
int i2 = i1 + stride;
int i3 = i2 + stride;

float D0 = d_Data[i0];
float D1 = d_Data[i1];
float D2 = d_Data[i2];
float D3 = d_Data[i3];

float T;
T = D0;
D0        = D0 + D2;
D2        = T - D2;
T = D1;
D1        = D1 + D3;
D3        = T - D3;
T = D0;
d_Data[i0] = D0 + D1;
d_Data[i1] = T - D1;
T = D2;
d_Data[i2] = D2 + D3;
d_Data[i3] = T - D3;
}
}
#ifdef DEBUG
#pragma omp target update from (d_Data[0:dataN])
for (int i = 0; i < dataN; i++) printf("k5 %f\n", d_Data[i]);
#endif


#pragma omp target teams num_teams(M) thread_limit(N/4)
{
float s_data[2048];
#pragma omp parallel 
{
int lid = omp_get_thread_num();
int gid = omp_get_team_num();
int gsz = omp_get_num_threads(); 

const int    N = 1 << log2N;
const int base = gid << log2N;

const float *d_Src = d_Data  + base;
float *d_Dst = d_Data + base;

for (int pos = lid; pos < N; pos += gsz)
{
s_data[pos] = d_Src[pos];
}

const int pos = lid;

for (int stride = N >> 2; stride > 0; stride >>= 2)
{
int lo = pos & (stride - 1);
int i0 = ((pos - lo) << 2) + lo;
int i1 = i0 + stride;
int i2 = i1 + stride;
int i3 = i2 + stride;

#pragma omp barrier
float D0 = s_data[i0];
float D1 = s_data[i1];
float D2 = s_data[i2];
float D3 = s_data[i3];

float T;
T = D0;
D0         = D0 + D2;
D2         = T - D2;
T = D1;
D1         = D1 + D3;
D3         = T - D3;
T = D0;
s_data[i0] = D0 + D1;
s_data[i1] = T - D1;
T = D2;
s_data[i2] = D2 + D3;
s_data[i3] = T - D3;
}

if (log2N & 1)
{
#pragma omp barrier

for (int pos = lid; pos < N / 2; pos += gsz)
{
int i0 = pos << 1;
int i1 = i0 + 1;

float D0 = s_data[i0];
float D1 = s_data[i1];
s_data[i0] = D0 + D1;
s_data[i1] = D0 - D1;
}
}

#pragma omp barrier

for (int pos = lid; pos < N; pos += gsz)
{
d_Dst[pos] = s_data[pos];
}
}
}
#ifdef DEBUG
#pragma omp target update from (d_Data[0:dataN])
for (int i = 0; i < dataN; i++) printf("k6 %f\n", d_Data[i]);
#endif

}

printf("Reading back GPU results...\n");
#pragma omp target update from (d_Data[0:dataN])
}

printf("Running straightforward CPU dyadic convolution...\n");
dyadicConvolutionCPU(h_ResultCPU, h_Data, h_Kernel, log2Data, log2Kernel);

printf("Comparing the results...\n");
sum_delta2 = 0;
sum_ref2   = 0;

for (i = 0; i < dataN; i++)
{
delta       = h_ResultCPU[i] - d_Data[i];
ref         = h_ResultCPU[i];
sum_delta2 += delta * delta;
sum_ref2   += ref * ref;
}

L2norm = sqrt(sum_delta2 / sum_ref2);

printf("Shutting down...\n");
free(h_ResultCPU);
free(h_Data);
free(h_Kernel);
free(d_Data);
free(d_Kernel);

printf("L2 norm: %E\n", L2norm);
printf(L2norm < 1e-6 ? "Test passed\n" : "Test failed!\n");
}
