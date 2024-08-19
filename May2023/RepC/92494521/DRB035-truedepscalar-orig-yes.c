#include <stdlib.h>
#include <stdio.h>
int main(int argc, char* argv[])
{ 
int i; 
int tmp;
tmp = 10;
int len=100;
int a[100];
#pragma omp parallel for
for (i=0;i<len;i++)
{ 
a[i] = tmp;
tmp =a[i]+i;
}
printf("a[50]=%d\n", a[50]);
return 0;      
}
