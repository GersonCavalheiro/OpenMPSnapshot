#include<stdio.h>
#include<omp.h>
int main()
{
int i,j,count=0;
#pragma omp parallel for collapse(2)
for(i=0;i<2;i++)
{
for(j=0;j<5;j++)
{
printf("thread=%d\n",omp_get_thread_num());
}
}
}
