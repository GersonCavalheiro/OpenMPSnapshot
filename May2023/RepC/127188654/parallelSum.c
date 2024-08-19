#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
int main(int argc, char* argv[])
{
int thread_count=10;
const int n=100;
int x[n];
for(int i=0;i<n;i++)
{
x[i]=1;
}
int sum=0;
printf("within %d threads and vector of size %d each element of value 1 \n",thread_count,n);
#pragma omp parallel for num_threads(thread_count) reduction(+:sum) schedule (static,2)
for(int i=0; i<n;i++)
{
sum +=x[i];
}
printf("sum = %d",sum);
printf("\n");
return 0;
}
