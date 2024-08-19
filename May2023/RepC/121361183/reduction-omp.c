#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#define dim 10
int main(int argc, char **argv)
{
int vec[dim];
int i,sum,thread;
thread=atoi(argv[1]);
for(i=0;i<dim;i++) vec[i]=i;
omp_set_num_threads(thread);
sum=0.0;
#pragma omp parallel for reduction(+:sum)
for(i=0;i<dim;i++)
sum=sum+vec[i];
printf("sum=%d\n",sum);
}