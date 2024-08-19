#include<stdio.h>
#include<omp.h>
int main()
{
int array1[5]={1,2,3,4,5},array2[5]={6,7,8,9,10},i,j;
#pragma omp parallel
{
#pragma omp for
for(i=0;i<5;i++)
{
array1[i]=array1[i]+array2[i];
}
}
for(i=0;i<5;i++)
{
printf("%d\n",array1[i]);
}
return 0;
}
