#include<stdio.h>
int main(int argc, char* argv[])
{
int i;
int len=100;
int a[len], b[len];
for (i=0;i<len;i++)
{  a[i]=i; b[i]=i;} 
#pragma omp parallel 
{
static int tmp;
#pragma omp for
for (i=0;i<len;i++)
{
tmp = a[i]+i;
a[i] = tmp;
}
}
#pragma omp parallel 
{
int tmp;
#pragma omp for
for (i=0;i<len;i++)
{
tmp = b[i]+i;
b[i] = tmp;
}
}
printf("a[50]=%d b[50]=%d\n", a[50], b[50]);
return 0;
}
