#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>	
#define N 12
int main() 
{
int i;
omp_set_num_threads(3);
#pragma omp parallel 
{
#pragma omp for schedule(static)
for (i=0; i < N; i++) {
int id=omp_get_thread_num();
printf("Loop 1: (%d) gets iteration %d\n",id,i);	
}
#pragma omp for schedule(static, 2)
for (i=0; i < N; i++) {
int id=omp_get_thread_num();
printf("Loop 2: (%d) gets iteration %d\n",id,i);	
}
#pragma omp for schedule(dynamic,2)
for (i=0; i < N; i++) {
int id=omp_get_thread_num();
printf("Loop 3: (%d) gets iteration %d\n",id,i);	
}
#pragma omp for schedule(guided,2)
for (i=0; i < N; i++) {
int id=omp_get_thread_num();
printf("Loop 4: (%d) gets iteration %d\n",id,i);	
}
}
return 0;
}
