#include <stdio.h>
#include <omp.h>
#define NUM_THREADS 1
int main( ) 
{
int tid = 0; 
omp_set_num_threads(NUM_THREADS);
printf("%d \n", omp_get_num_threads());
int x = 100000; 
int max = x - 1 ;
if(0 == tid) {
while(x % max != 0 ) {
max --;
}
} 
return 0;
}
