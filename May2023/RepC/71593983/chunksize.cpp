#include<iostream>
#include<omp.h>
int main()
{
int N =100;
#pragma omp parallel for schedule(dynamic,1)	
for(int i = 0; i < N; i++)
{
printf("Thread %i, itr = %i \n", omp_get_thread_num(), i);
}
return 0;
}
