#include <stdlib.h>
#include <stdio.h>
int main(int argc, char* argv[])
{
int i;
int len=100;
if (argc>1)
len = atoi(argv[1]);
int a[len];
for (i=0;i<len;i++)
a[i]=i;
#pragma omp parallel for
for (i=0;i<len-1;i++)
a[i+1]=a[i]+1;
return 0;
}
