#include <omp.h>
#include <stdio.h>
int main(void)
{    
omp_set_dynamic(0);
#pragma omp parallel num_threads(20)  
printf("Thread=%d\n",omp_get_thread_num());
return 0;
}