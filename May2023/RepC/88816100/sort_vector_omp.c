#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define ARRAY_SIZE 100000 
#define N_ARRAYS  10 
#define DEFAULT_THREADS 4
int cmpfunc (const void * a, const void * b){
return ( *(int*)a - *(int*)b );
}
int main(int argc,char **argv){
int threads, task, thread_id, i, j;
double t1, t2; 
if (argc == 1){
threads = DEFAULT_THREADS; 
}else if (argc == 2){
threads = atoi(argv[1]);
} else {
printf("Usage: %s number_of_threads.\n", argv[0]);
return 0;
}
printf("Run with %d threads...\n", threads);
t1 = omp_get_wtime(); 
int (*bag_of_tasks)[ARRAY_SIZE] = malloc (N_ARRAYS * sizeof *bag_of_tasks);
for (i = 0; i < N_ARRAYS; i++){
for(j = 0; j < ARRAY_SIZE; j++){
bag_of_tasks [i][j] = (ARRAY_SIZE-j)*(i+1);
}
}
omp_set_num_threads(threads);
#pragma omp parallel private (task, i)
#pragma omp for schedule (dynamic)
for (task = 0; task < N_ARRAYS; task++){
qsort(bag_of_tasks[task], ARRAY_SIZE, sizeof(int), cmpfunc);
}
t2 = omp_get_wtime(); 
printf("Tempo de execução: %3.2f segundos \n", t2-t1);
#if DEBUG == 1
printf("Print enabled...\n");
for (i = 0; i < N_ARRAYS; i++){
printf("Vector %d [", i);
for(j=0; j<ARRAY_SIZE; j++){
printf("%d ", bag_of_tasks[i][j]);
}
printf("]\n");
}
#endif
free(bag_of_tasks);
}
