#include <iostream>
#include <omp.h>
int main(void) {
int i;
#pragma omp parallel private(i)
{
#pragma omp for
for (i = 0; i < 10; i++) {
printf("#%d: i = %d\n", omp_get_thread_num(), i);
}
}
return EXIT_SUCCESS;
}
