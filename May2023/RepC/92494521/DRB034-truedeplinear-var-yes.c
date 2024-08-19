#include <stdlib.h>
int main(int argc, char* argv[])
{
int i;
int len=2000;
if (argc>1)
len = atoi(argv[1]);
int a[len];
for (i=0; i<len; i++)
a[i]=i; 
#pragma omp parallel for
for (i=0;i<len/2;i++)
a[2*i+1]=a[i]+1;
return 0;
}
