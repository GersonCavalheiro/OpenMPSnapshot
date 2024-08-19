#include<stdio.h>
#include<time.h>
#include<omp.h>
int main()
{
clock_t t1;
int a[1000000];
int sum=0;
for (int i=0;i<100000;i++)
{
a[i+1]=2*a[i];
}
t1 = clock();
for (int i=0;i<100000;i++)
{
sum=sum+a[i];
}
t1 = clock() - t1;
double time_taken1 = ((double)t1)/CLOCKS_PER_SEC;
printf("%f seconds for serial\n", time_taken1);
clock_t t2;
sum=0;
t2 = clock();
#pragma omp parallel
{
for (int j=0;j<100000;j++)
{
sum=sum+a[j];
}
t2 = clock() - t2;
double time_taken2 = ((double)t2)/CLOCKS_PER_SEC;
printf("%f seconds for parallel\n", time_taken2);	
}
return 0;
}
