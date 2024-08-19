#include <stdio.h>
#include <omp.h>
#define NUM_THREADS 4
#define N 100
int shared_counter; 
int main(int argc, char *argv[]) {
omp_set_num_threads(NUM_THREADS); 
shared_counter = 0; 
#pragma omp parallel shared(shared_counter)
{
int id = omp_get_thread_num(); 
int i;
for (i = 0; i < N; i++) {
#pragma omp critical
{
shared_counter++; 
}
}
printf("Thread %d: shared_counter = %d\n", id, shared_counter);
}
return 0;
}
