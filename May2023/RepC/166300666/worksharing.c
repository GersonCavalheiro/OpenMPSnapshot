#include <stdio.h>
#include "omp.h"
int main(void) {
int tid;
#pragma omp parallel private(tid)
{
tid = omp_get_thread_num();
if (tid == 0)
puts("I am the master");
else 
printf("I am worker %d\n", tid);
}
return 0;
}
