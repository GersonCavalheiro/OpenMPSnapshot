#include<stdio.h>
#include<omp.h>
int main()
{
int n = 5;
int i;
int prod = 2;
#pragma omp parallel
{
#pragma omp for reduction(*:prod)	
for(i = 1; i <= n; i++)
{
prod = prod*i;
#ifdef _OPENMP
printf("Thread %i at itr %i has value %i \n",
omp_get_thread_num(), i, prod);
#endif
}
printf("prod = %i \n ", prod);
}
return 0;
}
