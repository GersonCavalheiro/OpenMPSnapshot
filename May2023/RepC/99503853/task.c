#include<stdio.h>
#include<omp.h>
int main()
{
#pragma omp parallel
{
printf("1thread = %d\n",omp_get_thread_num());
#pragma omp single
{
printf("2thread = %d\n",omp_get_thread_num());
#pragma omp task
{
printf("3thread = %d\n",omp_get_thread_num());
int x = 0;
#pragma omp task
{
printf("4thread = %d\n",omp_get_thread_num());
x++;
printf("x = %d\n",x);
}
}
}
}
return 0;
}
