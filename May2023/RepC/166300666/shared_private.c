#include <stdio.h>
#include "omp.h"
int main(void) {
int shared_a, a = 7, tid;
#pragma omp parallel private(tid) shared(shared_a) firstprivate(a) 
{
tid = omp_get_thread_num();
printf("ID = %d --> Artimdan once a = %d\n", tid, a);
a++;
printf("ID = %d --> Artimdan sonra a = %d\n", tid, a);
#pragma omp master
shared_a = a;
}
printf("Shared a = %d\n", shared_a);
return 0;
}
