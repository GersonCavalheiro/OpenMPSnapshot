#include <stdlib.h>
#include <stdio.h>
int main(int argc, char* argv[])
{
int i;
int len=100;
int a[100];
for (i=0;i<len;i++)
a[i]=i;
#pragma omp parallel for
for (i=0;i<len-1;i++)
a[i+1]=a[i]+1;
printf("a[50]=%d\n", a[50]);   
return 0;
}
