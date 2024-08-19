#include<stdio.h>
#include<omp.h>
int main()
{
printf("Hello world!!\n");
#pragma omp parallel
{
#ifdef _OPENMP
printf("Hello from core %i \n", omp_get_thread_num() );
#endif
}
return 0;
}
