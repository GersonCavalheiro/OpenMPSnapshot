#include <stdio.h>
#include <omp.h>
#define N 20
#define NUM_THREADS 4
int main() 
{
int i;
#pragma omp parallel num_threads(NUM_THREADS) private(i) 
{
int id=omp_get_thread_num();
for (i=id; i < N; i=i+NUM_THREADS) {
printf("Thread ID %d Iter %d\n",id,i);	
}
}
return 0;
}
