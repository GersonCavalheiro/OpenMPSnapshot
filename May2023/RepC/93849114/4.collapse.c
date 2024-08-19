#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>	
#define N 5
int main() 
{
int i,j;
omp_set_num_threads(8);
#pragma omp parallel for private(j) 
for (i=0; i < N; i++) {
for (j=0; j < N; j++) {
int id=omp_get_thread_num();
printf("(%d) Iter (%d %d)\n",id,i,j);	
}
}
return 0;
}
