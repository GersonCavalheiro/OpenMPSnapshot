#include <omp.h>
#include <stdio.h>
int main(){
#pragma omp parallel	
{
printf("Hello, I am thread number %d out of %d threads\n", omp_get_thread_num(), omp_get_num_threads());
}
return 0;
}
