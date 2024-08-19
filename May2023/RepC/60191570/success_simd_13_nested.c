#include <stdio.h>
void foo(float *A, float *B, float *C)
{
int i, j;
#pragma omp simd
for(i=0; i<100; i++)
{
int m = 5;
if (i < 20)
m = 10;
int k = m;
for (j = 0; j < 10; j+=1)
{
B[i] += A[k] + C[j];
}
}
#pragma omp simd
for(i=0; i<100; i++)
{
int m = 5;
if (i < 20)
m = 10;
int k = m;
for (j = 0; j < 10; j+=1)
{
A[k] = B[i] + C[j];
}
}
}
void foo_sc(float *A, float *B, float *C)
{
int i, j;
for(i=0; i<100; i++)
{
int m = 5;
if (i < 20)
m = 10;
int k = m;
for (j = 0; j < 10; j++)
{
B[i] += A[k] + C[j];
}
}
for(i=0; i<100; i++)
{
int m = 5;
if (i < 20)
m = 10;
int k = m;
for (j = 0; j < 10; j++)
{
A[k] = B[i] + C[j];
}
}
}
int main()
{
float __attribute__((__aligned__(64))) a[500];
float __attribute__((__aligned__(64))) b[500];
float __attribute__((__aligned__(64))) c[500];
float __attribute__((__aligned__(64))) a_sc[500];
float __attribute__((__aligned__(64))) b_sc[500];
float __attribute__((__aligned__(64))) c_sc[500];
int i;
for (i=0; i<500; i++)
{
a[i] = i;
b[i] = i+20;
c[i] = i-30;
a_sc[i] = i;
b_sc[i] = i+20;
c_sc[i] = i-30;
}
foo(a, b, c);
foo_sc(a_sc, b_sc, c_sc);
for (i=0; i<500; i++)
{
if(a[i] != a_sc[i])
{
printf("A: %f != %f\n", a[i], a_sc[i]);
return 1;
}
if(b[i] != b_sc[i])
{
printf("B: %f != %f\n", b[i], b_sc[i]);
return 1;
}
if(c[i] != c_sc[i])
{
printf("C: %f != %f\n", c[i], c_sc[i]);
return 1;
}
}
printf("SUCCESS\n");
return 0;
}
