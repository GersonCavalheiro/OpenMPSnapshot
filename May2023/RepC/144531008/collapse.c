#include<stdio.h>
#include<omp.h>
void main(){
int i,j,n,a[50][50];
double t1,t2;
printf("Enter the value of n: ");
scanf("%d",&n);
t1=omp_get_wtime();
#pragma omp parallel num_threads(4)
{
int id=omp_get_thread_num();
#pragma omp for collapse(2)
for(i=0;i<n;i++)
for(j=0;j<n;j++)
{
a[i][j]=i+j;
printf("a[%d][%d] is %d\n",i,j,a[i][j]);
}
}
t2=omp_get_wtime();
printf("Time taken is %f\n",t2-t1);
}