#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h> 
#include <omp.h>
#define TOTAL 500000000
int main(int argc, char* argv[]){
if (argc!=2) return 1;
const int threads_num=atoi(argv[1]);
long sum=0;
#pragma omp parallel  num_threads(threads_num)
{
long localsum=0;
unsigned int seed=time(NULL);
#pragma omp for
for(long i=0;i<TOTAL;i++){	      			
float x = rand_r(&seed)/ (float) RAND_MAX;
float y = rand_r(&seed)/ (float) RAND_MAX;
if (sqrt((x*x) + (y*y)) < 1 )
localsum++;
}
#pragma omp atomic
sum+=localsum;			
}
}
