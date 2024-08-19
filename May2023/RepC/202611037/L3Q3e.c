#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
int main(int argc, char *argv[]){
omp_set_num_threads(4);
int i;
#pragma omp parallel for 
for(i = 0; i < 8; i++)
printf("eu %d, fiz %d\n", omp_get_thread_num(), i);
return 0;
}