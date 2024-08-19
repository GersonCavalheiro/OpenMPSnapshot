#include <stdio.h>
int a[100];
int main()
{
int len=100; 
int i,x=10;
#pragma omp parallel for 
for (i=0;i<len;i++)
{
a[i] = x;
x=i;
}
printf("x=%d",x);    
return 0;
} 
