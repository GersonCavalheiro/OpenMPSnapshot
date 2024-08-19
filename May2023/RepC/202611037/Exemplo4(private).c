#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
int main(int argc, char *argv[]) {
int i, n = 5;
int a;
(void) omp_set_num_threads(3);
#pragma omp parallel for private(i,a)
for (i=0; i<n; i++)
{
a = i+1;
printf("Thread %d tem um valor de a = %d para i = %d\n",
omp_get_thread_num(),a,i);
} 
return(0);
}