#include <math.h>
#include <malloc.h>
#include <stdlib.h>
int main (int argc, char* argv[])
{
const int width = 40;
const int height = 40;
int* input;
if(posix_memalign((void **) &input, 64, 1024*sizeof(int)) != 0)
{
exit(1);
}
int i;
for (i=0; i<1024; i++)
{
input[i] = -5;
}
#pragma omp simd
for (i=0; i<5; i++)
{
input[i] = i*i;
}
for (i=0; i<5; i++)
{
if (input[i] != i*i)
{
printf("1st fails!!\n");
return 1;
}
}
for (; i<32; i++)
{
if (input[i] != -5)
{
printf("2nd fails!!\n");
return 1;
}
}
#pragma omp simd
for (i=0; i<16; i++)
{
input[i] = i;
}
for (i=0; i<16; i++)
{
if (input[i] != i)
{
printf("3rd fails!!\n");
return 1;
}
}
for (; i<32; i++)
{
if (input[i] != -5)
{
printf("4th fails!!\n");
return 1;
}
}
#pragma omp simd
for (i=0; i<17; i++)
{
input[i] = -i;
}
for (i=0; i<17; i++)
{
if (input[i] != -i)
{
printf("5th fails!!\n");
return 1;
}
}
for (; i<32; i++)
{
if (input[i] != -5)
{
printf("6th fails!!\n");
return 1;
}
}
#pragma omp simd
for (i=0; i<18; i++)
{
input[i] = i*i;
}
for (i=0; i<18; i++)
{
if (input[i] != i*i)
{
printf("7th fails!!\n");
return 1;
}
}
for (; i<32; i++)
{
if (input[i] != -5)
{
printf("8th fails!!\n");
return 1;
}
}
for (i=0; i<1024; i++)
{
input[i] = -5;
}
int j;
for(j=0; j<16; j++)
{
#pragma omp simd
for (i=0; i<j; i++)
{
input[i] = -i-i;
}
for (i=0; i<j; i++)
{
if (input[i] != -i-i)
{
printf("1* fails!!\n");
return 1;
}
}
for (; i<32; i++)
{
if (input[i] != -5)
{
printf("2* fails!!\n");
return 1;
}
}
}
return 0;
}
