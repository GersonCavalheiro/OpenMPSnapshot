#include<omp.h>
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
int main(int argc,char *argv){
int seed = time(NULL);
float r;
#pragma omp parallel firstprivate(seed,r)
{
seed += omp_get_thread_num();
r = (float)rand_r(&seed) / (float)RAND_MAX;
printf("r = %f\n",r);
}
return 0;
}
