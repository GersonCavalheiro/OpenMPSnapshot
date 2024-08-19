#include<stdio.h>
#include<omp.h>
int main()
{
int i,j,arr[15]={3,4,1,56,45,234,90,67,2343,76,0,45,23,23,9},tmp;
#pragma omp parallel 
{
#pragma omp for ordered
for(i=0;i<14;i++)
{
#pragma omp ordered
for(j=i+1;j<15;j++)
{
if(arr[i]>arr[j])
{
tmp=arr[i];
arr[i]=arr[j];
arr[j]=tmp;
}
}
}	
}
for(i=0;i<15;i++)
{
printf("%d\t",arr[i]);
}
printf("\n");
return 0;
}
