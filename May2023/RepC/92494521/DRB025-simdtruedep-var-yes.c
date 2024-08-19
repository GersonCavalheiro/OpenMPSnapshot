#include <stdlib.h>
int main(int argc, char* argv[])
{
int i;
int len=100;
if (argc>1)
len = atoi(argv[1]);
int a[len], b[len];
for (i=0;i<len;i++)
{
a[i]=i;
b[i]=i+1;
}
#pragma omp simd
for (i=0;i<len-1;i++)
a[i+1]=a[i]*b[i];
return 0;
}
