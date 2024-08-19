#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#include<sys/time.h>
#include"temps1.c"
#define dim 1000
double mata[dim][dim],matb[dim][dim],matc[dim][dim];
int main(int argc,char **argv)
{
int i,j,count;
struct timeval t1,t2;
for(i=0;i<dim;i++)
for(j=0;j<dim;j++)
{
mata[i][j]=rand();
matb[i][j]=rand();
}
for(count=0;count<10;count++)
{
gettimeofday(&t1,NULL);
#pragma omp parallel for private(j)
for(i=0;i<dim;i++)
for(j=0;j<dim;j++) matc[i][j]=mata[i][j]+matb[i][j];
gettimeofday(&t2,NULL);
temps1(t1,t2);
}
}
