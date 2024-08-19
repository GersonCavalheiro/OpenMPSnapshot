#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#define VECTOR_SIZE 64
extern double t0,dt1,dt2;
long int random(void);
inline int min(int x,int y)
{
return x<y?x:y;
}
void __attribute__((noinline)) structfac_simd(int na, int nr, 
float* __attribute__((aligned(64))) a, 
float* __attribute__((aligned(64))) h, 
float* __attribute__((aligned(64))) E,
int DIM2_A, int DIM2_H, int DIM2_E)
{
int i;
float twopi;
twopi = 6.28318584f;
float f2 = 0.0f;
#pragma omp simd reduction(+:f2) aligned(a:64)
for (i=0; i<na; i++)
{
float t = a[i];
f2 += t*t;
}
f2 = 1.0f/sqrtf(f2);
#pragma omp simd aligned(a, h, E:64) suitable(DIM2_A, DIM2_H, DIM2_E)
for (i=0; i<nr; i++)
{
float A=0.0f;
float B=0.0f;
int j; 
for (j=0; j<na; j++) {
float arg = twopi*(h[i             ] * a[j + DIM2_A    ] +
h[i + DIM2_H    ] * a[j + (DIM2_A*2)] +
h[i + (2*DIM2_H)] * a[j + (DIM2_A*3)]);
A += a[j]*cosf(arg);
B += a[j]*sinf(arg);
}
E[i         ] = A*f2;
E[i + DIM2_E] = B*f2;
}
}
void __attribute__((noinline)) structfac_sc(int na, int nr, 
float* a, 
float* h, 
float* E,
int DIM2_A, int DIM2_H, int DIM2_E)
{
int i;
float twopi;
twopi = 6.28318584f;
float f2 = 0.0f;
for (i=0; i<na; i++)
f2 += a[i]*a[i];
f2 = 1.0f/sqrtf(f2);
for (i=0; i<nr; i++)
{
float A=0.0f;
float B=0.0f;
int j; 
for (j=0; j<na; j++)
{
float arg = twopi*(h[i             ] * a[j + DIM2_A    ] +
h[i + DIM2_H    ] * a[j + (DIM2_A*2)] +
h[i + (DIM2_H*2)] * a[j + (DIM2_A*3)]);
A += a[j]*cosf(arg);
B += a[j]*sinf(arg);
}
E[i         ] = A*f2;
E[i + DIM2_E] = B*f2;
}
}
void deta(int na, float*a, int DIM2_A)
{
int i,j;
#pragma novector
for (i=0; i<na; i++) {
if ( i & 1 )
a[i] = 6.0f;
else
a[i] = 7.0f;
for (j=1; j<DIM2_A; j++)
a[DIM2_A*j+i] = (float)random()/(float)RAND_MAX;
}
}
void deth(int nr, float*h, int DIM2_H)
{
const int hmax=20;
const int kmax=30;
const int lmax=15;
int i;
#pragma novector
for (i=0; i<nr; i++) {
h[i]            = rintf(2*hmax*(float)random()/(float)RAND_MAX - hmax);
h[DIM2_H + i]   = rintf(2*kmax*(float)random()/(float)RAND_MAX - kmax);
h[DIM2_H*2 + i] = rintf(2*lmax*(float)random()/(float)RAND_MAX - lmax);
}
}
double sumdif(float*a, float*b, int n)
{
double sum = 0.0;
int i;
#pragma novector
for (i=0; i<n; i++)
sum += fabsf(a[i] - b[i]);
return sum;
}
int main(int argc, char*argv[])
{
int na=100;   
int nr=100; 
int times=1;
int DIM2_H = 128;
int DIM2_E = 128;
int DIM2_A = 128;
int compute_serial = 0;
float *h;  
float *E;  
float *E1; 
float *a;  
int i;
if (posix_memalign((void **)&h, VECTOR_SIZE, sizeof(*h)*DIM2_H*nr) != 0)
{
return 1;
}
if (posix_memalign((void **)&E, VECTOR_SIZE, sizeof(*E)*DIM2_E*nr) != 0)
{
return 1;
}
if (posix_memalign((void **)&E1, VECTOR_SIZE, sizeof(*E1)*DIM2_E*nr) != 0)
{
return 1;
}
if (posix_memalign((void **)&a, VECTOR_SIZE, sizeof(*a)*DIM2_A*na) != 0)
{
return 1;
}
#pragma novector
for (i=0; i<DIM2_E*nr; i++)
E1[i] = E[i] = 0.0f;
deta(na,a, DIM2_A);
deth(nr,h, DIM2_H);
int tt;
#pragma novector
for (tt=0; tt<times; tt++)
{
structfac_simd(na,nr,a,h,E,DIM2_E,DIM2_H,DIM2_E);
}
structfac_sc(na,nr,a,h,E1,DIM2_A,DIM2_H,DIM2_E);
double sumdf=sumdif(E,E1, DIM2_E*nr);
if (sumdf >= 0.0001)
return 1;
return 0;
}
