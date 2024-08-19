#include<stdio.h>
#include<omp.h>
int main()
{
int i,j,k,a=5;
#pragma omp parallel
{
#pragma omp for
for(i=0;i<5;i++)
{
printf("ADD %d\n",a+i);
}
#pragma omp for
for(i=0;i<5;i++)
{
printf("SUB %d\n",a-i);
}
#pragma omp for 
for(i=0;i<5;i++)
{
printf("MUL %d\n",a*i);
}
}
}
