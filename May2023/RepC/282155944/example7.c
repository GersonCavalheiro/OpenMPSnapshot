#include <stdio.h>
#include <omp.h>
int main(void) {
int val = 0;
#pragma omp parallel num_threads(4) reduction(+: val)
{
val = omp_get_thread_num() * 10;
}
printf("Sum: %d\n", val);
return 0;
}
