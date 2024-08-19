#include<stdio.h>
#include<assert.h>
void f1(int *q)
{
#pragma omp critical
*q = 1;
#pragma omp flush
}
int main()
{ 
int i=0, sum=0; 
#pragma omp parallel reduction(+:sum) num_threads(10) 
{
f1(&i);
sum+=i;
}
assert (sum==10);
printf("sum=%d\n", sum);
return 0;   
}
