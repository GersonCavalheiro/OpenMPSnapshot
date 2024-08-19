#include <stdio.h>
#include <omp.h>
void do_some_work()
{
}
int main()
{
int N= 10, i= -1;
#pragma omp parallel num_threads(11)
{
int id= omp_get_thread_num();
#pragma omp for
for(i= 0; i < N; i++)
printf("Thread %d: %d. Address of i= %p\n",id,i,&i);
printf("Thread %d till %d. Address of i= %p\n",id,i,&i);
}
}
