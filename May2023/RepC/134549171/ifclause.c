#include <stdio.h>
#include <omp.h>
int main ()
{
int n=10,TID;
omp_set_num_threads(4);
#pragma omp parellel if(n>5) num_threads(n) default(none) private(TID) shared(n)
{
TID = omp_get_thread_num();
#pragma omp single
{
printf("Value of n = %d\n",n);
printf("Number of threads in parallel region: %d\n",omp_get_num_threads());
}
printf("Print statement executed by thread %d\n",TID);
}
return 0;
}
