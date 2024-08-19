#include<stdio.h>
#include<omp.h>
int main()
{
int a[5],i;
#pragma omp parallel
{
#pragma omp for
for(i=0;i<5;i++)
{	
a[i]=i*i;
printf("thread id: %d\n",omp_get_thread_num());
}
#pragma omp master
for(i=0;i<5;i++)
printf("a[%d]=%d thread id= %d\n",i,a[i],omp_get_thread_num());
printf("Just before barrier thread id: %d\n",omp_get_thread_num());
#pragma omp barrier
#pragma omp for
for(i=0;i<5;i++)
{	
a[i]+=i;
printf("thread id: %d\n", omp_get_thread_num());
}	
}
}
