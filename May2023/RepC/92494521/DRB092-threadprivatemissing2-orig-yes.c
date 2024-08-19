#include <stdio.h>
#include <assert.h>
int sum0=0, sum1=0;
int main()
{
int i, sum=0;
#pragma omp parallel
{
#pragma omp for
for (i=1;i<=1000;i++)
{
sum0=sum0+i;
}   
#pragma omp critical
{
sum= sum+sum0;
} 
}  
for (i=1;i<=1000;i++)
{
sum1=sum1+i;
}
printf("sum=%d; sum1=%d\n",sum,sum1);
return 0;
}
