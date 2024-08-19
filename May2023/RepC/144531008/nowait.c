#include<stdio.h>
#include<omp.h>
void main(){
int i,n;
double t1,t2;
printf("Enter value of n: ");
scanf("%d",&n);
t1=omp_get_wtime();
#pragma omp parallel num_threads(4)
{
int id=omp_get_thread_num();
#pragma omp for nowait
for(i=0;i<n;i++)
printf("Thread %d: value of i : %d\n",id,i);
printf("\nI am Thread %d NO WAIT EFFECT\n",id);
}
t2=omp_get_wtime();
printf("Time taken is %f\n",t2-t1);
}