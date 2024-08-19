#include <stdio.h>
#include <assert.h>
int sum0=0, sum1=0;
#pragma omp threadprivate(sum0)
void foo (int i)
{
sum0=sum0+i;
}
int main()
{
int len=1000;
int i, sum=0;
#pragma omp parallel copyin(sum0)
{
#pragma omp for
for (i=0;i<len;i++)
{
foo (i);
}   
#pragma omp critical
{
sum= sum+sum0;
} 
}  
for (i=0;i<len;i++)
{
sum1=sum1+i;
}
printf("sum=%d; sum1=%d\n",sum,sum1);
assert(sum==sum1);
return 0;
}
