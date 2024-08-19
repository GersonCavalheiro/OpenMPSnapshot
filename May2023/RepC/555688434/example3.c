#include <omp.h>
#include <stdio.h>
int tid;
#pragma omp threadprivate(tid)
int main() {
omp_set_num_threads(4);
int numt;
#pragma omp parallel default(shared)
{
tid = omp_get_thread_num();
if(tid==0){
for(int j=0; j<100000000; j++);
numt = omp_get_num_threads();
} 
}
#pragma omp parallel default(shared)
{
printf("Thread ID: %d of %d\n", tid, numt);
}
}