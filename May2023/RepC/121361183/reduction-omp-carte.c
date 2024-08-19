#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#define dim 10
int main(int argc, char **argv)
{
int vec[dim];
int i,sum,thread,sum_part;
thread=atoi(argv[1]);
for(i=0;i<dim;i++) vec[i]=i;
omp_set_num_threads(thread);
sum=0.0;
#pragma omp parallel private(sum_part)
{
sum_part=0;
#pragma omp for
for(i=0;i<dim;i++)
sum_part+=vec[i];
#pragma omp critical
sum=sum+sum_part;
}
printf("sum=%d\n",sum);
}