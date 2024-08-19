#include<stdio.h>
#include<omp.h>
int main()
{	
int a[100] = {}, b[100] = {}, c[100] = {}, id = 0;
#pragma omp parallel num_threads(10)
{
id = omp_get_thread_num();
#pragma omp for
for(int i = 0; i < 100; i++)
{
a[i] = i;
}
#pragma omp for
for(int j = 0; j < 100; j++)
{
b[j] = j;
}		 
}
}
