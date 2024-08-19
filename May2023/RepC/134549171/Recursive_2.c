#include<stdio.h>
#include<omp.h>
#include<stdlib.h>
#include<time.h>
void rec(int a);
void funcA(int a);
void funcB(int a);
void funcA(int a){
int TID = omp_get_thread_num();
rec(a-1);
}
void funcB(int a){
int TID = omp_get_thread_num();
rec(a-2);
}
void rec(int a)
{
if(a<0)
return;
#pragma omp task
funcA(a);
#pragma omp task
funcB(a);
#pragma omp taskwait
return;
}
int main(int argc, char *argv[]){
if(argc < 2){
printf("Usage ./a.out <number_of_threads>\n");
exit(1);
}
int nthreads;
unsigned int thread_qty = atoi(argv[1]);
omp_set_num_threads(thread_qty);
int i;
double start_time, run_time;
srand(time(NULL));
#pragma omp parallel
{
#pragma omp single nowait
{
start_time = omp_get_wtime();
rec(20);
run_time = omp_get_wtime() - start_time;
printf("%f\n",run_time);
}
}
return 0;
}
