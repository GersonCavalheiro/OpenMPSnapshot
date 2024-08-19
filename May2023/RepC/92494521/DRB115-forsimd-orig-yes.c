#include <stdio.h>
int main(int argc, char* argv[])
{
int i;
int len=100;
int a[100], b[100];
for (i=0;i<len;i++)
{
a[i]=i;
b[i]=i+1;
}
#pragma omp parallel for simd 
for (i=0;i<len-1;i++)
a[i+1]=a[i]+b[i];
printf("a[50]=%d\n",a[50]);
return 0;
}
