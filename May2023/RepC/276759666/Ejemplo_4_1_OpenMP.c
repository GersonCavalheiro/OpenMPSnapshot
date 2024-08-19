#include<omp.h>
#include<stdio.h>
#include<stdlib.h>
#include <unistd.h>
int main(int argc,char *argv){
float start = omp_get_wtime();
int tid;
#pragma omp parallel private(tid)
{ 
tid = omp_get_thread_num();
if (tid < omp_get_num_threads()/2 ) 
sleep(3);
printf("Hilo %d antes de la barrera en %f s\n",tid,omp_get_wtime()-start);
#pragma omp barrier
printf("Hilo %d después de la barrera en %f s\n",tid,omp_get_wtime()-start);
} 
return 0;
}
