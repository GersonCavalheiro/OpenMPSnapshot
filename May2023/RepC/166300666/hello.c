#include <stdio.h>
#include "omp.h"
int main(void) {
int tid, size;
#pragma omp parallel private(tid) shared(size)
{
tid = omp_get_thread_num();
size = omp_get_num_threads();
printf("%d of %d --> Hello World\n", tid, size);
}
return 0;
}
