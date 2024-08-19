#include <stdio.h>
#include <omp.h>
int main(void) {
int i;
#pragma omp parallel for num_threads(2)
for(i = 0; i < 4; i++) {
printf("Iteration: %d, Thread %d\n", i, omp_get_thread_num());
}
return 0;
}