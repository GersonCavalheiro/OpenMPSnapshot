#include<omp.h>
#include<stdio.h>
#include<stdlib.h>
int main(int argc,char *argv){
int i;
#pragma omp parallel for
for(i=0;i<100;i++)
printf("i = %d en hilo %d\n",i,omp_get_thread_num());
return 0;
}
