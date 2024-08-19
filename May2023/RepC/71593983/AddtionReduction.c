#include<stdio.h>
#include<omp.h>
int main()
{
double asum = 0.0;
int i, n = 5;
#pragma omp parallel
{
#pragma omp for reduction(+:asum) 
for(i = 0; i < n; i++)
{
asum = asum + i;
printf("Thread %i 's at itr %i has value of asum = %f \n"
,omp_get_thread_num(),i,asum);
}
}
printf("asum = %f \n", asum);
}
