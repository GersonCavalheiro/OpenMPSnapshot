#include <stdio.h>
#include <omp.h>
int
main(void)
{
int a,i,n=1000000;
#pragma omp master
{
a=0;
}
omp_set_num_threads(4);
#pragma omp parallel shared(a,n) private(i)
{
#pragma omp for reduction(+:a)
for(i=0;i<n;i++)
a+=1;
#pragma omp single nowait
printf("Valor de a:%d en el thread %d\n",a,omp_get_thread_num());
}
printf("La suma es %d\n",a);
return 0;
}
