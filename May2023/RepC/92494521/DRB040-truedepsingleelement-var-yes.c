#include <stdlib.h>
int main (int argc, char* argv[])
{
int len=1000;
int i; 
if (argc>1)
len = atoi(argv[1]);
int a[len];
a[0] = 2;
#pragma omp parallel for
for (i=0;i<len;i++)
a[i]=a[i]+a[0];
return 0;
}
