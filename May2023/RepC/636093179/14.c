#include <stdio.h>
#include <omp.h>
void do_some_work()
{
}
int main()
{
int N= 10000;
#pragma omp parallel
{
int threads= -1, id= -1, i= -1, start= -1, end= -1;
threads= omp_get_num_threads();
id= omp_get_thread_num();
start= 0;
end= N;
for(i= start; i < end; i++)
do_some_work();
printf("Thread %d worked from %d to %d\n",id,start,end);
}
}
