#include <omp.h>
#include <stdio.h>
int main() {
omp_set_num_threads(4);
int numt;
#pragma omp parallel default(shared)
{
int tid;
#pragma omp single nowait
{
for(int j=0; j<100000000; j++);
numt = omp_get_num_threads();
}
tid = omp_get_thread_num();
printf("Thread ID: %d of %d\n", tid, numt);
}
}