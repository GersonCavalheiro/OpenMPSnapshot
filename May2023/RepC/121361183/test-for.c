#include<omp.h>
#include<stdio.h>
int main(void)
{
int nthreads,tid;
int i,j;
int a[10][10];
omp_set_num_threads(10);
#pragma omp parallel private(i,tid)
{
tid=omp_get_thread_num();
printf("tid=%d ",tid);fflush(stdout);
#pragma omp parallel for
{
for(i=0;i<10;i++)
a[tid][i]=tid;
}
}
printf("\n");fflush(stdout);
for(i=0;i<10;i++)
{
for(j=0;j<10;j++)
{
printf("%d ",a[i][j]); fflush(stdout);
}
printf("\n");fflush(stdout);
}
}