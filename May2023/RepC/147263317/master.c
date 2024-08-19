#include <omp.h>
#include <stdio.h>
int main() {
#pragma omp parallel num_threads(4)
{
#pragma omp master 
{
printf("yello, i'm the master thread\n");
printf("you doubt me? tid = %d\n", omp_get_thread_num());
}
printf("I'm\n");
printf("back\n");
}
return 0;
}
