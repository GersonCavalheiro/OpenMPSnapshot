#include<stdio.h>
#include<omp.h>
void main(){
int i,n,a[50],b[50],sum;
double t1,t2;
printf("Enter the value of n: ");
scanf("%d",&n);
t1=omp_get_wtime();
#pragma omp parallel num_threads(4)
{
int id=omp_get_thread_num();
#pragma omp for ordered reduction(+:sum)
for(i=0;i<n;i++)
{
printf("Thread %d: value of i : %d\n",id,i);
sum=sum+i;
#pragma omp ordered
{
b[i]=i+1;
printf("b[%d] value is %d in ORDER\n",i,b[i]);
}
}
}
t2=omp_get_wtime();
printf("Time taken is %f\n",t2-t1);
}