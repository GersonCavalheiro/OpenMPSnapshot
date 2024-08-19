#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char *argv[]) {
int sum = 0;
#pragma omp parallel
{
int tid = omp_get_thread_num();
int i;
#pragma omp single
printf("omp before - thread = %d\n", tid);
for (i = 0; i < 16; ++i)
printf("all for - thread = %d, i = %d\n", tid, i);
#pragma omp for
for (i = 0; i < 16; ++i)
printf("omp for - thread = %d, i = %d\n", tid, i);
#pragma omp master
printf("omp after - thread = %d\n", tid);
}
}
