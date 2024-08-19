#include <stdio.h>
#include <omp.h>
int main()
{
int i;
#pragma omp parallel for
for(i=0;i<5;i++)
{
printf("%d. Hello World\n",i);
}
int p=30;
#pragma omp parallel shared(p) 
{
printf("Thread =%d p=%d \n",omp_get_thread_num(),p);
p+=omp_get_num_threads();
printf("Value of P set by thread %d = %d\n",omp_get_thread_num(),p);
}
printf("Final value of p= %d\n",p );
}
