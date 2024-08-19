#include<stdio.h>
#include<omp.h>
int main()
{
int threads = 0, id = 0, i = 0, N = -1000000;
#pragma omp parallel num_threads(14809)
{
threads = omp_get_num_threads();
id = omp_get_thread_num();
N = threads;
#pragma omp for
for(i = 1000000; i > N; i--){
printf("I am: %d and i = %d\n", id, i);
} 
#pragma omp master
{
printf("Total threads = %d\n", threads);
printf("Thread id = %d\n", id); 
}
}
}
