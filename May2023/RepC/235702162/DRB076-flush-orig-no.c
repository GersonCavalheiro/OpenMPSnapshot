#include<stdio.h>
#include<assert.h>
void f1(int *q)
{
*q = 1;
}
int main()
{ 
int i=0, sum=0; 
#pragma omp parallel reduction(+:sum) num_threads(10) private(i)
{
f1(&i);
sum+= i; 
}
assert (sum==10);
printf("sum=%d\n", sum);
return 0;   
}
