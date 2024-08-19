#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
int main(int argc, char *argv[]){
omp_set_num_threads(4);
int i, a = 2;
#pragma omp parallel for reduction(*:a)
for(i = 0; i < 8; i++){
a += 2;
}
printf("eu %d, total da redução %d\n", omp_get_thread_num(), a);
return 0;
}