#include<stdio.h>
#include<omp.h>
int main()
{
int i=0,j=1,N,nextTerm;
printf("Enter the limit\n");
scanf("%d",&N);
#pragma omp parallel
{
#pragma omp for ordered
for(i=1;i<=N;i++)
{
#pragma omp ordered
printf("%d\t",i);
nextTerm=i+j;
i=j;
j=nextTerm;
}
}
printf("SUM=%d\n",nextTerm);
return 0;
}
