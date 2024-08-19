#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
int print(int * x, int n);
int main()
{
int n = 1000;
int i;
int sum = 0;
double * x = (double *) malloc(n*sizeof(int));
#ifdef _OPENMP
printf("Before parallel region Num threads = %i \n",
omp_get_num_threads() );
printf("Main thread = %i\n", omp_get_thread_num());
#endif
#pragma omp parallel shared(x)	
{	
#ifdef _OPENMP
printf("In parallel region: Num threads = %i \n",
omp_get_num_threads() );
#endif
#pragma omp for schedule(dynamic,8)
for(i = 0; i < n; i++)
{
#ifdef _OPENMP
printf("Thread %i at itr %i \n", 
omp_get_thread_num(), i);
#endif
x[i] = (double)i;
}
}
free(x);
return 0;
}
int print(int * x, int n)
{
int i;
for(i = 0; i < n; i++)
{
printf("%i \n", x[i]);
}
return 0;
}
