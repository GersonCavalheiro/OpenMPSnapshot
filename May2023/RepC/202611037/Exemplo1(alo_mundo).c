#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
int main (int argc, char *argv[]) {
int nthreads, tid;
#pragma omp parallel private(nthreads, tid)
{
tid = omp_get_thread_num();
printf("Alô mundo da thread = %d\n", tid);
if (tid == 0)  {
nthreads = omp_get_num_threads();
printf("Número de  threads = %d\n", nthreads);
}
} 
}
