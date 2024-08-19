#include <stdio.h>
#include <omp.h>
int main() {
int shared_var = 0;
int tid, local_var;
#pragma omp parallel private(local_var, tid)
{
tid = omp_get_thread_num();
local_var = tid + 1;
#pragma omp flush(shared_var)
#pragma omp critical
{
shared_var += local_var;
printf("Thread %d updated shared_var to %d\n", tid, shared_var);
}
#pragma omp flush(shared_var)
}
printf("Final value of shared_var is %d\n", shared_var);
return 0;
}
