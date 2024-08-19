#include <omp.h>
#include <stdio.h>

#define PARALLELISM_ENABLED 0

int main(){

printf("Hello from Sequential Region\n");
printf("Number of executing threads is: %d\n", omp_get_num_threads());

#pragma omp parallel if(PARALLELISM_ENABLED) num_threads(8)
{
if(omp_get_thread_num() == 0){
printf("Number of executing threads is: %d\n", omp_get_num_threads());
}

printf("Hello from thread %d\n", omp_get_thread_num());
}

printf("Hello from Sequential Region, again\n");
printf("Number of executing threads is: %d\n", omp_get_num_threads());

return 0;
}