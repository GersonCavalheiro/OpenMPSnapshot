#include<stdio.h>
#include<omp.h>
int main()
{
int i,a=40,b=20,c=30;
#pragma omp parallel
{
#pragma omp for nowait
for(i=0;i<4;i++)
{
a=a+1;
printf("a=%d by thread=%d\n",a,omp_get_thread_num());
}
#pragma omp for nowait
for(i=0;i<4;i++)
{
b=b-1;
printf("b=%d by thread=%d\n",b,omp_get_thread_num());
}
#pragma omp for nowait
for(i=0;i<4;i++)
{
c=c*2;
printf("c=%d by thread=%d\n",c,omp_get_thread_num());
}
}
}
