#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
void print(void);
int main(int argc, char* argv[])
{
int threads_num=5;
#pragma omp parallel num_threads(threads_num)
print();
return 0;
}
void print (void)
{
int my_rank=omp_get_thread_num();
int thread_count=omp_get_num_threads();
printf("Hello wolrd from thread %d of %d \n",my_rank,thread_count);
}