#include <stdio.h>
#include <stdlib.h>
#include <omp.h>   
int main(int argc, char* argv[]) {
int x = 5;
int thread_count = strtol(argv[1], NULL, 10); 
#pragma omp parallel num_threads(thread_count) private(x)
{
int my_rank = omp_get_thread_num();
printf("Thread %d > before initialization, x = %d\n", 
my_rank, x);
x = 2*my_rank + 2;
printf("Thread %d > after initialization, x = %d\n", 
my_rank, x);
}
printf("After parallel block, x = %d\n", x);
return 0; 
}  
