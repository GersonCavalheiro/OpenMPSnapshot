#include <stdio.h>
#include <omp.h>
void do_some_work()
{
}
int main()
{
int N= 10, i= -10;
#pragma omp parallel
{
int id= omp_get_thread_num();
#pragma omp for
for(i= 0; i < N; i++)
printf("Thread %d: %d\n",id,i);
printf("Thread %d till %d\n",id,i);
}
}
