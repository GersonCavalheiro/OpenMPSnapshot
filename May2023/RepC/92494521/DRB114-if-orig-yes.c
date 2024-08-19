#include <stdlib.h>
#include <stdio.h>
#include <time.h>
int main(int argc, char* argv[])
{
int i;
int len=100;
int a[100];
for (i=0;i<len;i++)
a[i]=i;
srand(time(NULL));
#pragma omp parallel for if (rand()%2)
for (i=0;i<len-1;i++)
a[i+1]=a[i]+1;
printf("a[50]=%d\n", a[50]);   
return 0;
}
