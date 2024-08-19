#include <stdlib.h>
#include <stdio.h>
int main(int argc, char* argv[])
{
int i;
int a[2000];
for (i=0; i<2000; i++)
a[i]=i; 
#pragma omp parallel for
for (i=0;i<1000;i++)
a[2*i+1]=a[i]+1;
printf("a[1001]=%d\n", a[1001]);  
return 0;
}
