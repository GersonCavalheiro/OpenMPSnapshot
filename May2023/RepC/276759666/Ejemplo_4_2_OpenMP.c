#include<omp.h>
#include<stdio.h>
#include<stdlib.h>
#include <unistd.h>
int main(int argc,char *argv){
float start = omp_get_wtime();
int tid,i;
#pragma omp parallel private(tid)
{ 
tid = omp_get_thread_num();
#pragma omp critical
{
printf("Hilo %d: Línea 1\n",tid);
if(tid==2) sleep(1);
if(tid==1) sleep(3);
printf("Hilo %d: Línea 2\n",tid);
if(tid==0) sleep(2);
printf("Hilo %d: Línea 3\n",tid);
}
} 
return 0;
}
