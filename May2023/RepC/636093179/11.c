#include <stdio.h>
#include <omp.h>
void fun(int id)
{
printf("%d: Entering fun...\n",id);
#pragma omp parallel
{
printf("%d is now %d\n",id,omp_get_thread_num());
}
printf("%d: Leaving fun...\n",id);
}
int main()
{
omp_set_num_threads(2);
omp_set_nested(1);
#pragma omp parallel
{
int threads= -1, id= -1, cores= -1;
threads= omp_get_num_threads();
id= omp_get_thread_num();
printf("Total threads= %d\n",threads);
printf("Thread id= %d\n",id);
fun(id);
}
}
