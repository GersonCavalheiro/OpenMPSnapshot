#include <stdio.h>
#include <omp.h>
int main(void) {
int i;
#pragma omp parallel num_threads(2)
{
#pragma omp for
for(i = 0; i < 4; i++){
printf("Loop: %d, Thread %d\n", i, omp_get_thread_num());
}
}
return 0;
}