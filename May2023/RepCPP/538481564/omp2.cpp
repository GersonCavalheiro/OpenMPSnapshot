#include <omp.h>
#include <stdio.h>

void execute(){

if (omp_get_max_threads() == 1)
return;

#pragma omp parallel if(omp_get_max_threads() > 1)
{
printf("Hello World! from thread num %d of %d\n", omp_get_thread_num(), omp_get_num_threads());
}
}

int main() {

omp_set_num_threads(3);
printf("Executing with 3 threads:\n");
execute();

omp_set_num_threads(1);
printf("Executing with 1 thread:\n");
execute();
}
