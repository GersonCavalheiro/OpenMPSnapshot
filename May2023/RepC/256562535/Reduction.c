#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h> 
#include <omp.h>
#include <string.h>
#define TOTAL 500000000
long inc(){
unsigned int seed=time(NULL);
float x = rand_r(&seed)/ (float) RAND_MAX;
float y = rand_r(&seed)/ (float) RAND_MAX;
if (sqrt((x*x) + (y*y)) < 1 )
return 1;
return 0;
}
int main(int argc, char* argv[]){
if (argc!=2) return 1;
const int threads_num=atoi(argv[1]);
printf("Theads: %d\n",threads_num);
long sum=0;
#pragma omp parallel num_threads(threads_num)
{
#pragma omp for reduction(+:sum)
for(long i=0;i<TOTAL;i++){	      			
sum = inc()+sum;
}			
}
}
