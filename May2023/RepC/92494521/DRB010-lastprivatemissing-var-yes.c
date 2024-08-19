#include <stdio.h>
#include <stdlib.h>
int main(int argc, char* argv[])
{
int i,x;
int len = 10000;
if (argc>1)
len = atoi(argv[1]);
#pragma omp parallel for private (i) 
for (i=0;i<len;i++)
x=i;
printf("x=%d",x);
return 0;
}
